srun python3.6 main.py --config_file ./config/nii_config.yml trainer.gpus [0,1,2,3] dataset.slice_num 16 dataset.batch_size 100 trainer.logger.test_tube.name nii_resnext101_3d_1ch_s16
