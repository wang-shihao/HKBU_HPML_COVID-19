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

from config.config import add_config
import models
import losses
import data
import engine

from multiprocessing import Process
import time

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
        trainer = build_trainer(cfg, hparams)
        trainer.test(model)
    else:
        model = build_module(cfg)
        print(model)
        trainer = build_trainer(cfg, hparams)
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
    parent_parser.add_argument('--predict_only', action='store_true', help='if true run model(samples). Predict on the given samples.')
    parent_parser.add_argument( "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    # each LightningModule defines arguments relevant to it
    hparams = parent_parser.parse_args()
    assert not (hparams.test_only and hparams.predict_only), "You can't set both 'test_only' and 'predict_only' True"

    # ---------------------
    # LOG NVIDIA-SMI
    # ---------------------
    #logfile = './nvidia-smi.log'
    #class NVLogger(Process):
    #    def __init__(self, logfile='./nvidia-smi.log', interval=10):
    #        super().__init__()
    #        self.logfile = logfile
    #        self.interval = interval

    #    def run(self):
    #        while True:
    #            os.system('nvidia-smi >> {}'.format(self.logfile))
    #            time.sleep(self.interval)

    #p = NVLogger()
    #p.start()
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
