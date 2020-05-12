
from collections import OrderedDict
import torch
from torchline.engine import build_module, MODULE_REGISTRY, DefaultModule
from torchline.utils import topk_acc, AverageMeterGroup

__all__ = [
    'CTModule'
]

@MODULE_REGISTRY.register()
class CTModule(DefaultModule):
    def __init__(self, cfg):
        super(CTModule, self).__init__(cfg)
        h, w = self.cfg.input.size
        self.example_input_array = torch.rand(1, 3, 2, h, w)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        try:
            # forward pass
            inputs, gt_labels, paths = batch
            predictions = self.forward(inputs)

            # calculate loss
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
            self.print_log(batch_idx, True, inputs, self.train_meters)

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
            'valid_loss': loss_val,
            'valid_acc_1': val_acc_1,
            f'valid_acc_{self.cfg.topk[-1]}': val_acc_k,
        })
        tqdm_dict = {k: v for k, v in dict(output).items()}
        self.valid_meters.update({key: val.item() for key, val in tqdm_dict.items()})
        self.print_log(batch_idx, False, inputs, self.valid_meters)

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        tqdm_dict = {key: val.avg for key, val in self.valid_meters.meters.items()}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'valid_loss': self.valid_meters.meters['valid_loss'].avg}
        return result
