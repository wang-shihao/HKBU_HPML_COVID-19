srun -n 1 --cpus-per-task 20 python main.py 
--config_file ./config/config.yml 
trainer.gpus [0,1,2,3] model.name r2plus1d_18 model.model_depth 18 
dataset.slice_num 64 dataset.batch_size 64 
dataset.is_color False
model.n_input_channels 1
trainer.train_percent_check 0.2
trainer.logger.test_tube.name r2plus1d_18_datasub0.2_bs64_s64_c1