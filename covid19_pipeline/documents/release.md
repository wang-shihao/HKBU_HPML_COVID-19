
# 2020.05.19

- 新增通道为1的数据集和模型，通道数由`cfg.dataset.is_color`决定，`True`表示三通道,`False`表示单通道，即灰白图片。网络输入通道由`cfg.model.n_input_channels`决定
- 修复多GPU下重复打印的问题
