```shell
# 该目录包含使用OneFlow进行剪枝的代码。
# 该目录包含两个子目录：pruner和utils。
# pruner子目录包含不同剪枝算法和调度程序的代码。
# utils子目录包含计算神经网络中操作数量的代码。
# 根目录中的文件是子目录中代码的辅助文件。
# __init__.py文件是指示目录为Python包的空文件。
# importance.py文件包含计算神经网络中每个层的重要性的代码。
# ops.py文件包含卷积和线性层剪枝的代码。
# dependency.py文件包含检查OneFlow剪枝模块依赖项的代码。
.
├── dependency.py
├── _helpers.py
├── importance.py
├── __init__.py
├── ops.py
├── pruner
│   ├── algorithms
│   │   ├── batchnorm_scale_pruner.py
│   │   ├── group_norm_pruner.py
│   │   ├── __init__.py
│   │   ├── magnitude_based_pruner.py
│   │   ├── metapruner.py
│   │   └── scheduler.py
│   ├── function.py
│   └── __init__.py
└── utils
    ├── __init__.py
    ├── op_counter.py
    └── utils.py

4 directories, 16 files
```
