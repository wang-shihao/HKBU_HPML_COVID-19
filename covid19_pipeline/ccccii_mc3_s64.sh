#srun -N 1 -n 1 --gres=gpu:3 --nodelist=hkbugpusrv06 python3.6 main.py --config_file ./config/3sets_config.yml trainer.gpus [1,2,3] model.name mc3_18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name 3sets_mc3_18_bs9_s64
#~/py36/bin/python main.py --config_file ./config/ccccii_config.yml trainer.gpus [1] model.name mc3_18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name ccccii_mc3_18_bs9_s64
~/py36/bin/python main.py --test_only      --config_file ./config/3sets_config.yml trainer.gpus [1] model.name mc3_18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name 3sets_mc3_18_bs9_s64 predict_only.weights_path ./output/ccccii_mc3_18_bs9_s64/version_0/checkpoints/epoch=51-valid_acc_1=93.13.ckpt
#~/py36/bin/python main.py --test_only     --config_file ./config/ccccii_config.yml trainer.gpus [1] model.name mc3_18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name 3sets_mc3_18_bs9_s64 predict_only.weights_path ./output/ccccii_mc3_18_bs9_s64/version_0/checkpoints/epoch=51-valid_acc_1=93.13.ckpt
~/py36/bin/python main.py --test_only --config_file ./config/mosmeddata_config.yml trainer.gpus [1] model.name mc3_18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name 3sets_mc3_18_bs9_s64 predict_only.weights_path ./output/ccccii_mc3_18_bs9_s64/version_0/checkpoints/epoch=51-valid_acc_1=93.13.ckpt
~/py36/bin/python main.py --test_only --config_file ./config/covidctset_config.yml trainer.gpus [1] model.name mc3_18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name 3sets_mc3_18_bs9_s64 predict_only.weights_path ./output/ccccii_mc3_18_bs9_s64/version_0/checkpoints/epoch=51-valid_acc_1=93.13.ckpt