
日志文件夹命名规范： `model_data_imgsize_batchsize_slicenum_other`

例如`mc3_18_datas_img224_bs8_s64_tf_flip`表示：
- 模型为mc3_18
- 数据集为dataset_seg， datac表示dataset_clean
- 图片大小是224x224
- batch size是8
- slice数量是64
- tf表示数据增强操作，这里表示使用了flip

运行命令: `srun --cpus-per-task 2 python main.py trainer.gpus [0,1] trainer.logger.test_tube.name densent3d121_datas_img128_bs8_s64 model.name densenet3d model.model_depth 121`

# Exp1. 第一轮验证集上结果（测试不同模型）

训练集的slice是随机采样，验证集是先将所有slice读取后使用Centercrop操作。图片大小均为224x224。

|model|batch_size|slice_num|gpus|dataset|valid_acc1|transform|marks|
|---|---|---|---|---|---|---|---|
|mc3_18|8|64|2|clean|85.12| | |
|mc3_18|8|64|2|seg|83.75| | | |
|r3d_18|8|64|2|seg|79.62| | | |
|r2plus1d_18|8|64|2|seg|68.62| | | |
|mc3_18|4|90|4|seg|78.25| | | |

上述结果均是运行了100个epoch后的结果，比较而言，mc3_18比其他模型性能要好，而且slice设置为64效果也比90要好一些。另外cleaned数据集的结果也好于seg数据集，这个还有待研究具体原因，猜测可能是因为seg数据集因为都是切割后的数据，所以相似性会高一些，因此难度也就高一些了。

# Exp2. 测试Model,Slice的效果

训练集的slice是随机采样，验证集是等距采样，数据增强操作先resize，然后再centercrop。
图片大小固定为128.

## Model


|model|batch_size|slice_num|img_size|dataset|valid_acc1|transform|marks|
|---|---|---|---|---|---|---|---|
|mc3_18|8|64|128|seg|| | |
|r3d_18|8|64|128|seg|| | | |
|r2plus1d_18|8|64|128|seg|| | | |
|resnet101|8|64|128|seg|| | |
|densenet121|8|64|128|seg|| | |
|resnext|8|64|128|seg|| | |


## Slice 

|model|batch_size|slice_num|img_size|dataset|valid_acc1|transform|marks|
|---|---|---|---|---|---|---|---|
|mc3_18|8|16|128|seg|| | |
|mc3_18|8|32|128|seg|| | |
|mc3_18|8|64|128|seg|| | |
|mc3_18|8|90|128|seg|| | |
|mc3_18|8|128|128|seg|| | |


# 3. 测试不同transform效果

选择上面实验中最好的slice设置

|model|batch_size|slice_num|img_size|dataset|valid_acc1|transform|marks|
|---|---|---|---|---|---|---|---|
|mc3_18|8|x|64|seg|  | noise| |
|mc3_18|8|x|64|seg|  | swap| |
|mc3_18|8|x|64|seg|  | affine| |
|mc3_18|8|x|64|seg|  | blur| |


Todo
- [ ] 对比实验设计
  - [ ] 模型： img128, slice64 (mc3_18,r3d_18,r2plus1d_18,densenet,resnet101,resnext,densenet121,wide_resnet101)
  - [ ] slice: img128
  - [ ] 深度
  - [x] transform (后续)
- [ ] 统计每一类的acc
- [ ] 整合更多3D模型
