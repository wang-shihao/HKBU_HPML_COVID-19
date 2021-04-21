#################################################################
#train on homenode
#srun -N 1 -n 1 --gres=gpu:3 --nodelist=hkbugpusrv05 python3.6 main.py --config_file ./config/3sets_config.yml trainer.gpus [1,2,3] model.name resnet3d model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name 3sets_r3d_18_bs9_s64
#################################################################
#train
#~/py36/bin/python main.py --config_file ./config/ccccii_config.yml trainer.gpus [1] model.name resnet3d model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name ccccii_r3d_18_bs9_s64
#################################################################
#test
#~/py36/bin/python main.py   --test_only   --config_file ./config/3sets_config.yml  trainer.gpus [1] model.name resnet3d model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name 3sets_r3d_18_bs9_s64 predict_only.weights_path ./output/ccccii_r3d_18_bs9_s64/version_3/checkpoints/epoch=80-valid_acc_1=92.07.ckpt
#~/py36/bin/python main.py   --test_only   --config_file ./config/ccccii_config.yml trainer.gpus [1] model.name resnet3d model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name 3sets_r3d_18_bs9_s64 predict_only.weights_path ./output/ccccii_r3d_18_bs9_s64/version_3/checkpoints/epoch=80-valid_acc_1=92.07.ckpt 
#~/py36/bin/python main.py --test_only --config_file ./config/mosmeddata_config.yml trainer.gpus [1] model.name resnet3d model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name 3sets_r3d_18_bs9_s64 predict_only.weights_path ./output/ccccii_r3d_18_bs9_s64/version_3/checkpoints/epoch=80-valid_acc_1=92.07.ckpt
#~/py36/bin/python main.py --test_only --config_file ./config/covidctset_config.yml trainer.gpus [2] model.name resnet3d model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name 3sets_r3d_18_bs9_s64 predict_only.weights_path ./output/ccccii_r3d_18_bs9_s64/version_3/checkpoints/epoch=80-valid_acc_1=92.07.ckpt
#################################################################
#transfer
#nohup ~/py36/bin/python main.py --transfer --config_file ./config/ccccii_config.yml trainer.gpus [0] model.name r3d_18 model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name ccccii_r3d_18_bs9_s64 predict_only.weights_path ../../FedML-COVID19/fedml_experiments/distributed/fedavg/outputs/FedAVG\(d\)hetero-covid_ct-r100-e1-r3d_18-sgd-bs3-lr0.001-wd0.0001-layers64.pth > transfer_ccccii_123.log 2>&1 &
#sleep 15s
#nohup ~/py36/bin/python main.py --transfer --config_file ./config/ccccii_config.yml trainer.gpus [1] model.name r3d_18 model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name ccccii_r3d_18_bs9_s64 predict_only.weights_path ../../FedML-COVID19/fedml_experiments/distributed/fedavg/outputs12/FedAVG\(d\)hetero-covid_ct-r100-e1-r3d_18-sgd-bs3-lr0.001-wd0.0001-layers64.pth > transfer_ccccii_12.log 2>&1 &
#sleep 15s
#nohup ~/py36/bin/python main.py --transfer --config_file ./config/ccccii_config.yml trainer.gpus [2] model.name r3d_18 model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name ccccii_r3d_18_bs9_s64 predict_only.weights_path ../../FedML-COVID19/fedml_experiments/distributed/fedavg/outputs23/FedAVG\(d\)hetero-covid_ct-r100-e1-r3d_18-sgd-bs3-lr0.001-wd0.0001-layers64.pth > transfer_ccccii_23.log 2>&1 &
#sleep 15s
#nohup ~/py36/bin/python main.py --transfer --config_file ./config/ccccii_config.yml trainer.gpus [3] model.name r3d_18 model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name ccccii_r3d_18_bs9_s64 predict_only.weights_path ../../FedML-COVID19/fedml_experiments/distributed/fedavg/outputs31/FedAVG\(d\)hetero-covid_ct-r100-e1-r3d_18-sgd-bs3-lr0.001-wd0.0001-layers64.pth > transfer_ccccii_31.log 2>&1 &
#sleep 15s
#nohup ~/py36/bin/python main.py   --transfer  --config_file ./config/ccccii_config.yml  trainer.gpus [1] model.name resnet3d model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name ccccii_r3d_18_bs9_s64 predict_only.weights_path ./output/3sets_r3d_18_bs9_s64/version_3/checkpoints/epoch=76-valid_acc_1=85.20.ckpt > transfer_ccccii_distributed.log 2>&1 &
#sleep 15s
nohup ~/py36/bin/python main.py   --transfer  --config_file ./config/ccccii_config.yml  trainer.gpus [2] model.name resnet3d model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name ccccii_r3d_18_bs9_s64 predict_only.weights_path ./output/mosmeddata_r3d_18_bs9_s64/version_1/checkpoints/epoch=32-valid_acc_1=78.68.ckpt > transfer_ccccii_distributed_2.log 2>&1 &
#sleep 15s
nohup ~/py36/bin/python main.py   --transfer  --config_file ./config/ccccii_config.yml  trainer.gpus [3] model.name resnet3d model.model_depth 18 dataset.slice_num 64 dataset.batch_size 9 trainer.logger.test_tube.name ccccii_r3d_18_bs9_s64 predict_only.weights_path ./output/covidctset_r3d_18_bs9_s64/version_1/checkpoints/epoch=67-valid_acc_1=96.03.ckpt > transfer_ccccii_distributed_3.log 2>&1 &
