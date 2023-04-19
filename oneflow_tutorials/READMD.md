## 仓库OneFlow启动指南

## 环境准备

### 下载代码

```
git clone git@github.com:Oneflow-Inc/OneFlow-Pruning.git
```

### 安装依赖

```shell
# https://github.com/Oneflow-Inc/oneflow#install-with-pip-package
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117 # oneflow包

git checkout develop

pip install -r requirements.txt
```

### Pruning库介绍

在复杂的网络结构中，参数组之间可能存在依赖关系，需要同时进行剪枝。我们的工作通过提供自动机制来对参数进行分组，以便于加速它们的有效删除来解决这一挑战。具体而言，Torch-Pruning 通过使用虚拟输入转发您的模型，跟踪网络以建立图形，并记录层之间的依赖关系来完成此操作。当您剪枝单个层时，OneFlow-Pruning 通过返回 `tp.Group` 来识别和分组所有耦合层。此外，如果存在像 flow.split 或 flow.cat 这样的操作 ( 操作对应代码: OneFlow-Pruning/oneflow_pruning/dependency.py:665 )，所有剪枝索引将自动对齐。 



<div align="center">
<img src="https://user-images.githubusercontent.com/118866310/232971854-7ef76f29-5448-4395-b46b-e264a1a0df2f.png" width="100%">
</div>

使用 DepGraph，很容易设计一些“组级”标准来估计整个组的重要性，而不是单个层。在我们的论文中，我们设计了一个简单的 [GroupPruner](https://github.com/VainF/Torch-Pruning/blob/745f6d6bafba7432474421a8c1e5ce3aad25a5ef/torch_pruning/pruner/algorithms/group_norm_pruner.py#L8) (c) 来学习耦合层之间的一致稀疏性。


<div align="center">
<img src="https://user-images.githubusercontent.com/118866310/232971937-b8b20c47-a4df-43fd-a377-08c14a261cc5.png" width="80%">
</div>


图中画红框处为 OneFlow develop 文件
![image](https://user-images.githubusercontent.com/118866310/232967858-91c651f7-0830-43c6-9991-4584c313e8eb.png)

其中 jupyter-notebook文件:
- 0. QuickStart.ipynb ( some basic yet all-in-one examples to show the features of OneFlow-Pruning.  )
- 1. Customize Your Own Pruners
- 2. Exploring Dependency Groups

https://github.com/Oneflow-Inc/OneFlow-Pruning/tree/develop/oneflow_tutorials

## 可能存在问题

目前只是保证 catbackward  这个判段正确
OneFlow-pruning库里面 split 和 view 可能存在问题。 还未具体测试。

```py
// OneFlow-Pruning/oneflow_pruning/dependency.py:667

elif "catbackward" in grad_fn.name().lower():
    module = ops._ConcatOp()
elif "split" in grad_fn.name().lower():
    module = ops._SplitOp()
elif "view" in grad_fn.name().lower() or 'reshape' in grad_fn.name().lower():
    module = ops._ReshapeOp()
else:
    # treate other ops as element-wise ones, like Add, Sub, Div, Mul.
    module = ops._ElementWiseOp(grad_fn.name())
gradfn2module[grad_fn] = module
```

## 已知待修复问题
AttributeError: 'oneflow._oneflow_internal.FunctionNode' object has no attribute 'variable' => OneFlow-Pruning/oneflow_pruning/dependency.py:709

AttributeError: 'ConvTranspose2d' object has no attribute 'in_channels'

AttributeError: 'Conv2d' object has no attribute 'transposed' 

AttributeError: module 'oneflow.nn' has no attribute 'MultiheadAttention'

## 已修复
align AutoGrad engine function_node name and next_functions type with pytorch AutoGrad Engine  https://github.com/Oneflow-Inc/OneFlow-Pruning/issues/6
## test【-2023-04-18】
> 0.9.1.dev20230417+cu117
- oneflow_tutorials目录下.ipynb文件均可以正常执行
- oneflow_tests目录下测试 日志：log.txt 

