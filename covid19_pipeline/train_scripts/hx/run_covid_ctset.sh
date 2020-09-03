srun -n 1 --cpus-per-task 20 python main.py 
--config_file ./config/covid_ctset_config.yml
trainer.gpus [0,1,2,3] model.name densenet3d model.model_depth 121
dataset.slice_num 40 dataset.batch_size 16 
dataset.is_color False
model.n_input_channels 1
trainer.logger.test_tube.name densenet3d121_datact_bs16_s40