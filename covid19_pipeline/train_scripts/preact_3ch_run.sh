/usr/local/bin/python main.py --config_file ./config/preact_config.yml trainer.gpus [0,1,2,3] dataset.slice_num 64 dataset.batch_size 8 trainer.logger.test_tube.name pre_act_resnet101_3d_1ch trainer.resume_from_checkpoint ./output/pre_act_resnet101_3d_1ch/version_4/checkpoints/epoch=147-valid_acc_1=82.96.ckpt