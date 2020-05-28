import os
from collections import OrderedDict

import numpy as np
import torch
from sklearn import metrics

from torchline.engine import MODULE_REGISTRY, DefaultModule, build_module
from torchline.utils import AverageMeterGroup, topk_acc

from .utils import mixup_data, mixup_loss_fn

__all__ = [
    'CTModule'
]

@MODULE_REGISTRY.register()
class CTModule(DefaultModule):
    def __init__(self, cfg):
        super(CTModule, self).__init__(cfg)
        h, w = self.cfg.input.size
        self.example_input_array = torch.rand(1, 3, 2, h, w)
        self.crt_batch_idx = 0
        self.inputs = self.example_input_array

    def training_step_end(self, output):
        self.print_log(self.trainer.batch_idx, True, self.inputs, self.train_meters)
        return output

    def validation_step_end(self, output):
        self.crt_batch_idx += 1
        self.print_log(self.crt_batch_idx, False, self.inputs, self.valid_meters)
        return output

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        try:
            # forward pass
            inputs, gt_labels, paths = batch
            self.crt_batch_idx = batch_idx
            self.inputs = inputs
            if self.cfg.mixup.enable:
                inputs, gt_labels_a, gt_labels_b, lam = mixup_data(inputs, gt_labels, self.cfg.mixup.alpha)
                mixup_y = [gt_labels_a, gt_labels_b, lam]
            predictions = self.forward(inputs)

            # calculate loss
            if self.cfg.mixup.enable:
                loss_val = mixup_loss_fn(self.loss, predictions, *mixup_y)
            else:
                loss_val = self.loss(predictions, gt_labels)

            # acc
            acc_results = topk_acc(predictions, gt_labels, self.cfg.topk)
            tqdm_dict = {}

            if self.on_gpu:
                acc_results = [torch.tensor(x).to(loss_val.device.index) for x in acc_results]

            # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
            if self.trainer.use_dp or self.trainer.use_ddp2:
                loss_val = loss_val.unsqueeze(0)
                acc_results = [x.unsqueeze(0) for x in acc_results]

            tqdm_dict['train_loss'] = loss_val
            for i, k in enumerate(self.cfg.topk):
                tqdm_dict[f'train_acc_{k}'] = acc_results[i]

            output = OrderedDict({
                'loss': loss_val,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            self.train_meters.update({key: val.item() for key, val in tqdm_dict.items()})

            # can also return just a scalar instead of a dict (return loss_val)
            return output
        except Exception as e:
            print(str(e))
            print(batch_idx, paths)
            pass

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        inputs, gt_labels, paths = batch
        self.inputs = inputs
        predictions = self.forward(inputs)

        loss_val = self.loss(predictions, gt_labels)

        # acc
        val_acc_1, val_acc_k = topk_acc(predictions, gt_labels, self.cfg.topk)

        if self.on_gpu:
            val_acc_1 = val_acc_1.cuda(loss_val.device.index)
            val_acc_k = val_acc_k.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc_1 = val_acc_1.unsqueeze(0)
            val_acc_k = val_acc_k.unsqueeze(0)
        
        output = OrderedDict({
            'valid_loss': torch.tensor(loss_val),
            'valid_acc_1': torch.tensor(val_acc_1),
            f'valid_acc_{self.cfg.topk[-1]}': val_acc_k,
        })
        tqdm_dict = {k: v for k, v in dict(output).items()}
        self.valid_meters.update({key: val.item() for key, val in tqdm_dict.items()})
        # self.print_log(batch_idx, False, inputs, self.valid_meters)

        if self.cfg.module.analyze_result:
            output.update({
                'predictions': predictions.detach(),
                'gt_labels': gt_labels.detach(),
            })
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        self.crt_batch_idx = 0
        tqdm_dict = {key: val.avg for key, val in self.valid_meters.meters.items()}
        valid_loss = torch.tensor(self.valid_meters.meters['valid_loss'].avg)
        valid_acc_1 = torch.tensor(self.valid_meters.meters['valid_acc_1'].avg)
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict,
                  'valid_loss': valid_loss,
                  'valid_acc_1': valid_acc_1}

        if self.cfg.module.analyze_result:
            predictions = []
            gt_labels = []
            for output in outputs:
                predictions.append(output['predictions'])
                gt_labels.append(output['gt_labels'])
            predictions = torch.cat(predictions)
            gt_labels = torch.cat(gt_labels)
            analyze_result = self.analyze_result(gt_labels, predictions)
            self.log_info(analyze_result)
            result.update({'analyze_result': analyze_result, 'predictions': predictions, 'gt_labels': gt_labels})
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        predictions = result['predictions'].cpu().detach().numpy()
        gt_labels = result['gt_labels'].cpu().detach().numpy()
        path = self.cfg.log.path
        np.save(os.path.join(path,'predictions.npy'), predictions)
        np.save(os.path.join(path,'gt_labels.npy'), gt_labels)
        result = {key:val for key, val in result.items() if key not in ['predictions', 'gt_labels']}
        return result

    def analyze_result(self, gt_labels, predictions):
        '''
        Args:
            gt_lables: tensor (N)
            predictions: tensor (N*C)
        '''
        return str(metrics.classification_report(gt_labels.cpu(), predictions.cpu().argmax(1), digits=4))
