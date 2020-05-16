# 一、注册机制解释

本框架使用注册机制，因此可以在命令行或者yml文件中修改参数值即可自动调用不同模型、损失函数等一切设置。下面以定义一个新的模型为例来解释注册机制。

通过观察`models/model.py`可以看到模型的注册是通过使用装饰器实现的，而模型的装饰器是`@META_ARCH_REGISTRY`,该装饰器会自动将函数名进行注册，例如下面代码中将`mc3_18`模型进行注册，后续只需要将`model.name`设置为`mc3_18`即可调用该模型。

另外装饰器所注册的函数只能接收一个参数，即`cfg`。这个`cfg`可以理解成是一个全局的参数字典，保存了你所定义的所有参数设置(`config.yml`)，之后你只需要将你需要的参数从`cfg`中解析出来，然后返回你要的模型即可。

```python
@META_ARCH_REGISTRY.register()
def mc3_18(cfg):
    return generate_model(cfg, 'mc3_18')
```


# 二、如何定义一个新的模型

1. 新建一个文件`models/new_model.py`

```python
import torch.nn as nn
from torchline.models import META_ARCH_REGISTRY

__all__ = [
    'NewModel',
    '_NewModel'
]

class _NewModel(nn.Module):
    def __init__(self, num_classes):
        super(_NewModel, self).__init__()
        self.linear = nn.Linear(10, num_classes)
    
    def forward(self, x):
        return self.linear(x)

@META_ARCH_REGISTRY.register()
def NewModel(cfg):
    num_classes = cfg.model.num_classes 
    # 你需要确保`config/config.py`或者`config/config.yml`中有`model.num_classes`这个参数
    return _NewModel(num_classes)
```

2. 更新`models/__init__.py`

```
from .new_model import *
```

3. 更新`config/config.py`

假设上面的`cfg.model.num_classes`是一个还未定义的参数，则你需要修改`config/config.py`的`add_config`函数

```python
def add_config(cfg):
    ...
    
    cfg.model.num_classes = 10

    ...
    return cfg
```