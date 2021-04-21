"""
Runs a model on a single node across N-gpus.
"""
import argparse
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torchline.config import get_cfg
from torchline.engine import build_module
from torchline.models import META_ARCH_REGISTRY
from torchline.trainer import build_trainer
from torchline.utils import Logger
from collections import OrderedDict
import torch.nn as nn

from config.config import add_config
import utils
import models
import losses
import data
import engine


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """

    cfg = get_cfg()
    cfg = add_config(cfg)
    cfg.setup_cfg_with_hparams(hparams)
    if hasattr(hparams, "test_only") and hparams.test_only:
        model = build_module(cfg)
        checkpoint = torch.load(cfg.predict_only.weights_path)
        #print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'])
        trainer = build_trainer(cfg, hparams)
        trainer.test(model)
    elif hasattr(hparams, "cam_only") and hparams.cam_only:
        model = build_module(cfg)
        cam = utils.CAM3D(cfg, model)
        cam.run()
    else:
        model = build_module(cfg)
        trainer = build_trainer(cfg, hparams)
        if hasattr(hparams, 'transfer') and hparams.transfer:
            checkpoint = torch.load(cfg.predict_only.weights_path)
            if cfg.predict_only.weights_path.endswith("ckpt"):
                model.load_state_dict(checkpoint['state_dict'])
                for param in model.parameters():
                    param.require_grad = False
                num_ftrs = model.model.fc.in_features
                model.model.fc = nn.Linear(num_ftrs, 3)
            else:
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    #name = k[7:] # remove `module.`
                    name = 'model.'+k
                    new_state_dict[name] = v
                #print(model)    
                model.load_state_dict(new_state_dict)
                for param in model.parameters():
                    param.require_grad = False
                num_ftrs = model.model.fc.in_features
                model.model.fc = nn.Linear(num_ftrs, 3)
            print("#############load weights successful#############")

        trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument("--config_file", default="./config/config.yml", metavar="FILE", help="path to config file")
    parent_parser.add_argument('--test_only', action='store_true', help='if true, return trainer.test(model). Validates only the test set')
    parent_parser.add_argument('--cam_only', action='store_true', help='if true, save heatmap by CAM')
    parent_parser.add_argument('--predict_only', action='store_true', help='if true run model(samples). Predict on the given samples.')
    parent_parser.add_argument('--transfer', action='store_true', help='if true then do transfer learning from cfg.predict_only.weights.')
    parent_parser.add_argument( "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    # each LightningModule defines arguments relevant to it
    hparams = parent_parser.parse_args()
    assert not (hparams.test_only and hparams.predict_only), "You can't set both 'test_only' and 'predict_only' True"

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
