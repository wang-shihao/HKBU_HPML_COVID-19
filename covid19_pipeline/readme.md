# Dependencies
- python3.6
- pytorch(>=1.3.0)
- torchline==0.3.0.4

# English

```
python main.py --config_file ./config/config.yml model.name mc3_18 trainer.logger.test_tube.name experiemnt1 trainer.gpus [0,1]
```

The above command will first read the default settings defined in `./config/config.yml`, then the latter configurations will update the value. For example,  `model.name mc3_18` will specify to load `mc3_18` model. You can also specify other model names that are defined in `models` module.

There are two types of configurations:
- one is set by `argparse`. 
  - You should specify `--` if you use this type of configurations, e.g., '--config_file'.
  - You can refer other cfy in `main.py`
- the other is those defined in `yml` files. 
  - Nearly all cfgs are set in `yml` files, such as batch size. learning rate, and model, etc. 


# 中文

运行方法


```bash
python main.py --config_file ./config/config.yml model.name mc3_18 trainer.logger.test_tube.name experiemnt1 trainer.gpus [0,1]
```

上述命令会读取`./config/config.yml`中设置的默认的参数，后面的命令会更新对应参数的值，比如`model.name mc3_18`就是将模型改为`mc3_18`，你也可以设置其他名字，这些名字需要在`models`模块中提前定义好。

运行参数分为两种
- 一种是由`argparse`设定的参数（参见main.py)，这类参数必须要有`--`指定，比如`--config_file`可以指定读取某个yml文件。
- 另一种是定义在`yml`文件中的参数。基本上实验所有参数都在yml文件中定义好了，比如batch size, learning rate, 使用何种模型(model.name)等等。如果你需要定义一个新的参数，则需要在`config/config.py/add_config`函数内新增加参数名。另外对于已经定义好的参数，如果你想修改，有三种方法可以灵活修改这类参数，这些方法按照由高到低的优先级排序为
  - `config/config.py`
  - `config/config.yml`
  - `命令行`

