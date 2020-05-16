# run 

示例：

```bash
srun python main.py --config_file ./config/config.yml model.name mc3_18 trainer.logger.test_tube.name experiemnt1 trainer.gpus [0,1]
```

运行参数分为两种，一种是由`argparse`设定的参数（参见main.py)，另一种是定义在`yml`文件中的参数。第二种参数设置可以在三个地方灵活修改，不过参数读取的顺序是`config/config.py`->`config/config.yml`->`命令行`。如果你需要定义一个新的参数，则需要在`config/config.py/add_config`函数内新增加参数名。

上述命令的意思是：
- `--config_file`:指定读取yml文件,这个是由`argpaser`指定的参数
- 其他参数定义不需要加`--`。
  - `model.name`: 指定模型，上面例子会自动调用`mc3_18`模型
  - `trainer.logger.test_tube.name`: 本项目使用`test_tube`库帮助日志存储，该参数会指定日志存储文件夹的名字，比如会自动创建`./output/experiemnt1/`,所有的日志回报存在这个路径。
  - `trainer.gpus`: 指定使用的GPU数量，只能接收一个`list`
  - 其他参数可以查看`config/config.yml`文件
