srun python main.py --config_file ./config/config.yml trainer.gpus [0,1,2,3] dataset.slice_num 128 dataset.batch_size 32 trainer.logger.test_tube.name resnext101_3d_1ch_s128
