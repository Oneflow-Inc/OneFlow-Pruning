文档通过 chatGpt [data-2023-04-07] 翻译 ，原文链接:

<div align="center"> <h1>Torch-Pruning <br> <h3>Towards Any Structural Pruning<h3> </h1> </div>
<div align="center">
<img src="assets/intro.png" width="45%">
</div>

Torch-Pruning (TP) 是一个通用的库，可以对各种神经网络进行结构剪枝，包括 **Vision Transformers、Yolov7、FasterRCNN、SSD、ResNet、DenseNet、ConvNext、RegNet、ResNext、FCN、DeepLab、VGG** 等。与 [torch.nn.utils.prune](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) 不同，Torch-Pruning 使用称为 DepGraph 的（非深度）图算法从模型中物理删除耦合的参数（通道）。要探索更多可剪枝模型，请参阅 [benchmarks/prunability](benchmarks/prunability)。到目前为止，TP 兼容 Torchvision 0.13.1 中的 **73/85=85.8%** 模型。在这个 repo 中，一个关于实用结构剪枝的 [资源列表](practical_structural_pruning.md) 正在不断更新中。

有关更多技术细节，请参阅论文：



> [**DepGraph: Towards Any Structural Pruning**](https://arxiv.org/abs/2301.12900)   
> [Gongfan Fang](https://fangggf.github.io/), [Xinyin Ma](https://horseee.github.io/), [Mingli Song](https://person.zju.edu.cn/en/msong), [Michael Bi Mi](https://dblp.org/pid/317/0937.html), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)   

请不要犹豫，如果您在使用该库时遇到任何问题或有任何与论文相关的问题，请打开[讨论](https://github.com/VainF/Torch-Pruning/discussions)或[问题](https://github.com/VainF/Torch-Pruning/issues)。我们很乐意帮助您解决任何问题。 


### **特性:**
- [x] 适用于各种神经网络的结构（通道）剪枝，包括[卷积神经网络](benchmarks/prunability/torchvision_pruning.py#L19)（例如ResNet、DenseNet、Deeplab）、[Transformer](benchmarks/prunability/torchvision_pruning.py#L11)（例如ViT）和检测器（例如[Yolov7](benchmarks/prunability/yolov7_train_pruned.py#L102)、[FasterRCNN、SSD](benchmarks/prunability/torchvision_pruning.py#L92)）
- [x] 高级剪枝器：[MagnitudePruner](https://arxiv.org/abs/1608.08710)、[BNScalePruner](https://arxiv.org/abs/1708.06519)、[GroupPruner](https://arxiv.org/abs/2301.12900)（我们论文中使用的简单剪枝器）、RandomPruner等。
- [x] 计算图跟踪和依赖建模。
- [x] 支持的模块：Conv、Linear、BatchNorm、LayerNorm、Transposed Conv、PReLU、Embedding、MultiheadAttention、nn.Parameters和[自定义模块](tests/test_customized_layer.py)。
- [x] 支持的操作：split、concatenation、skip connection、flatten、reshape、view、所有逐元素操作等。
- [x] [低级剪枝函数](torch_pruning/pruner/function.py)
- [x] [基准测试](benchmarks)和[教程](tutorials)
- [x] 一个[资源列表](practical_structural_pruning.md)，用于实用的结构剪枝。
- [x] 自动剪枝未包含在任何标准层或操作中的未包装nn.Parameter。


### **计划:**

**我们有很多想法，但目前只有少数贡献者。我们希望吸引更多有才华的人加入我们，实现这些想法，使 Torch-Pruning 成为一个实用的库。**
- [ ] 一个 [Torchvision](https://pytorch.org/vision/stable/models.html) 兼容性基准测试 (**73/85=85.8**, :heavy_check_mark:) 和 [timm](https://github.com/huggingface/pytorch-image-models) 兼容性基准测试。
- [ ] 更多检测器（我们正在研究 YOLO 系列的剪枝，如 YOLOv7 :heavy_check_mark:，YOLOv8）
- [ ] 从头开始剪枝/初始化时剪枝。
- [ ] 语言、语音和生成模型。
- [ ] 更多高级剪枝器，如 [FisherPruner](https://arxiv.org/abs/2108.00708)、[GrowingReg](https://arxiv.org/abs/2012.09243) 等。
- [ ] 更多标准层：GroupNorm、InstanceNorm、Shuffle Layers 等。
- [ ] 更多 Transformer，如 Vision Transformer (:heavy_check_mark:)、Swin Transformer、PoolFormer。
- [ ] 块/层/深度剪枝
- [ ] CIFAR、ImageNet 和 COCO 的剪枝基准测试。


## Installation
```bash
pip install torch-pruning # v1.1.2
```
or
```bash
git clone https://github.com/VainF/Torch-Pruning.git # recommended
```

## 快速开始

这里提供了 Torch-Pruning 的快速入门。更详细的说明可以在 [tutorals](./tutorials/) 中找到。

### 0. 工作原理

在复杂的网络结构中，参数组之间可能存在依赖关系，需要同时进行剪枝。我们的工作通过提供自动机制来对参数进行分组，以便于加速它们的有效删除来解决这一挑战。具体而言，Torch-Pruning 通过使用虚拟输入转发您的模型，跟踪网络以建立图形，并记录层之间的依赖关系来完成此操作。当您剪枝单个层时，Torch-Pruning 通过返回 `tp.Group` 来识别和分组所有耦合层。此外，如果存在像 torch.split 或 torch.cat 这样的操作，所有剪枝索引将自动对齐。 



<div align="center">
<img src="assets/dep.png" width="100%">
</div>

使用 DepGraph，很容易设计一些“组级”标准来估计整个组的重要性，而不是单个层。在我们的论文中，我们设计了一个简单的 [GroupPruner](https://github.com/VainF/Torch-Pruning/blob/745f6d6bafba7432474421a8c1e5ce3aad25a5ef/torch_pruning/pruner/algorithms/group_norm_pruner.py#L8) (c) 来学习耦合层之间的一致稀疏性。


<div align="center">
<img src="assets/group_sparsity.png" width="80%">
</div>


### 1. A minimal example

```python
import torch
from torchvision.models import resnet18
import torch_pruning as tp

# 加载预训练的resnet18模型
model = resnet18(pretrained=True).eval()

# 1. 为resnet18构建依赖图
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. 指定要剪枝的通道。这里我们剪枝索引为[2, 6, 9]的通道。
pruning_idxs = [2, 6, 9]
pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )

# 打印剪枝组的详细信息
print(pruning_group.details())  # or print(pruning_group)

# 3. 剪枝所有与model.conv1耦合的分组层（包括model.conv1本身）。
if DG.check_pruning_group(pruning_group): # 避免完全剪枝，即通道数为0。
    pruning_group.prune()

# 4. 保存和加载剪枝后的模型
torch.save(model, 'model.pth') # 保存模型对象
model_loaded = torch.load('model.pth') # 不需要load_state_dict

```
  
这个例子演示了使用DepGraph的基本剪枝流程。请注意，resnet.conv1与几个层耦合。让我们打印出结果组并观察剪枝操作如何“触发”其他操作。在以下输出中，“A => B”表示剪枝操作“A”触发剪枝操作“B”。group[0]是由“DG.get_pruning_group”指定的剪枝根。


```
--------------------------------
          Pruning Group
--------------------------------
[0] prune_out_channels on conv1 (Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)) => prune_out_channels on conv1 (Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)), idxs=[2, 6, 9] (Pruning Root)
[1] prune_out_channels on conv1 (Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)) => prune_out_channels on bn1 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), idxs=[2, 6, 9]
[2] prune_out_channels on bn1 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => prune_out_channels on _ElementWiseOp(ReluBackward0), idxs=[2, 6, 9]
[3] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_out_channels on _ElementWiseOp(MaxPool2DWithIndicesBackward0), idxs=[2, 6, 9]
[4] prune_out_channels on _ElementWiseOp(MaxPool2DWithIndicesBackward0) => prune_out_channels on _ElementWiseOp(AddBackward0), idxs=[2, 6, 9]
[5] prune_out_channels on _ElementWiseOp(MaxPool2DWithIndicesBackward0) => prune_in_channels on layer1.0.conv1 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), idxs=[2, 6, 9]
[6] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on layer1.0.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), idxs=[2, 6, 9]
[7] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on _ElementWiseOp(ReluBackward0), idxs=[2, 6, 9]
[8] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_out_channels on _ElementWiseOp(AddBackward0), idxs=[2, 6, 9]
[9] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_in_channels on layer1.1.conv1 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), idxs=[2, 6, 9]
[10] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on layer1.1.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), idxs=[2, 6, 9]
[11] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on _ElementWiseOp(ReluBackward0), idxs=[2, 6, 9]
[12] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_in_channels on layer2.0.downsample.0 (Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)), idxs=[2, 6, 9]
[13] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_in_channels on layer2.0.conv1 (Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)), idxs=[2, 6, 9]
[14] prune_out_channels on layer1.1.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => prune_out_channels on layer1.1.conv2 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), idxs=[2, 6, 9]
[15] prune_out_channels on layer1.0.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => prune_out_channels on layer1.0.conv2 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), idxs=[2, 6, 9]
--------------------------------
```
For more details about grouping, please refer to [tutorials/2 - Exploring Dependency Groups](https://github.com/VainF/Torch-Pruning/blob/master/tutorials/2%20-%20Exploring%20Dependency%20Groups.ipynb)

#### How to scan all groups:
Just like what we do in the [MetaPruner](https://github.com/VainF/Torch-Pruning/blob/b607ae3aa61b9dafe19d2c2364f7e4984983afbf/torch_pruning/pruner/algorithms/metapruner.py#L197), one can use ``DG.get_all_groups(ignored_layers, root_module_types)`` to scan all groups sequentially. Each group will begin with a layer that matches a type in the "root_module_types" parameter. By default, these groups contain a full index list ``idxs=[0,1,2,3,...,K]`` that covers all prunable parameters. If you are intended to prune only partial channels/dimensions, you can use ``group.prune(idxs=idxs)``.

```python
for group in DG.get_all_groups(ignored_layers=[model.conv1], root_module_types=[nn.Conv2d, nn.Linear]):
    # handle groups in sequential order
    idxs = [2,4,6] # your pruning indices
    group.prune(idxs=idxs)
    print(group)
```



### 2. High-level Pruners

Leveraging the DependencyGraph, we developed several high-level pruners in this repository to facilitate effortless pruning. By specifying the desired channel sparsity, you can prune the entire model and fine-tune it using your own training code. For detailed information on this process, we encourage you to consult the [this tutorial](https://github.com/VainF/Torch-Pruning/blob/master/tutorials/1%20-%20Customize%20Your%20Own%20Pruners.ipynb). Additionally, you can find more practical examples in [benchmarks/main.py](benchmarks/main.py).

```python
import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True)

# Importance criteria
example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.MagnitudeImportance(p=2)

ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

iterative_steps = 5 # progressive pruning
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps,
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # finetune your model here
    # finetune(model)
    # ...
```

#### Sparse Training
Some pruners like [BNScalePruner](https://github.com/VainF/Torch-Pruning/blob/dd59921365d72acb2857d3d74f75c03e477060fb/torch_pruning/pruner/algorithms/batchnorm_scale_pruner.py#L45) and [GroupNormPruner](https://github.com/VainF/Torch-Pruning/blob/dd59921365d72acb2857d3d74f75c03e477060fb/torch_pruning/pruner/algorithms/group_norm_pruner.py#L53) require sparse training before pruning. This can be easily achieved by inserting just one line of code ``pruner.regularize(model)`` in your training script. The pruner will update the gradient of trainable parameters.
```python
for epoch in range(epochs):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, target)
        loss.backward()
        pruner.regularize(model) # <== for sparse learning
        optimizer.step()
```

#### Interactive Pruning
All high-level pruners support interactive pruning. You can use ``pruner.step(interactive=True)`` to get all groups and interactively prune them by calling ``group.prune()``. This feature is useful if you want to control/monitor the pruning process.

```python
for i in range(iterative_steps):
    for group in pruner.step(interactive=True): # Warning: groups must be handled sequentially. Do not keep them as a list.
        print(group) 
        # do whatever you like with the group 
        # ...
        group.prune() # you should manually call the group.prune()
        # group.prune(idxs=[0, 2, 6]) # you can even change the pruning behaviour with the idxs parameter
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # finetune your model here
    # finetune(model)
    # ...
```



### 3. Low-level pruning functions

While it is possible to manually prune your model using low-level functions, this approach can be quite laborious, as it requires careful management of the associated dependencies. As a result, we recommend utilizing the aforementioned high-level pruners to streamline the pruning process.

```python
tp.prune_conv_out_channels( model.conv1, idxs=[2,6,9] )

# fix the broken dependencies manually
tp.prune_batchnorm_out_channels( model.bn1, idxs=[2,6,9] )
tp.prune_conv_in_channels( model.layer2[0].conv1, idxs=[2,6,9] )
...
```

The following pruning functions are available:
```python
tp.prune_conv_out_channels,
tp.prune_conv_in_channels,
tp.prune_depthwise_conv_out_channels,
tp.prune_depthwise_conv_in_channels,
tp.prune_batchnorm_out_channels,
tp.prune_batchnorm_in_channels,
tp.prune_linear_out_channels,
tp.prune_linear_in_channels,
tp.prune_prelu_out_channels,
tp.prune_prelu_in_channels,
tp.prune_layernorm_out_channels,
tp.prune_layernorm_in_channels,
tp.prune_embedding_out_channels,
tp.prune_embedding_in_channels,
tp.prune_parameter_out_channels,
tp.prune_parameter_in_channels,
tp.prune_multihead_attention_out_channels,
tp.prune_multihead_attention_in_channels,
```

### 4. Customized Layers

Please refer to [tests/test_customized_layer.py](https://github.com/VainF/Torch-Pruning/blob/master/tests/test_customized_layer.py).

### 5. Benchmarks

Our results on {ResNet-56 / CIFAR-10 / 2.00x}

| Method | Base (%) | Pruned (%) | $\Delta$ Acc (%) | Speed Up |
|:--    |:--:  |:--:    |:--: |:--:      |
| NIPS [[1]](#1)  | -    | -      |-0.03 | 1.76x    |
| Geometric [[2]](#2) | 93.59 | 93.26 | -0.33 | 1.70x |
| Polar [[3]](#3)  | 93.80 | 93.83 | +0.03 |1.88x |
| CP  [[4]](#4)   | 92.80 | 91.80 | -1.00 |2.00x |
| AMC [[5]](#5)   | 92.80 | 91.90 | -0.90 |2.00x |
| HRank [[6]](#6) | 93.26 | 92.17 | -0.09 |2.00x |
| SFP  [[7]](#7)  | 93.59 | 93.36 | +0.23 |2.11x |
| ResRep [[8]](#8) | 93.71 | 93.71 | +0.00 |2.12x |
||
| Ours-L1 | 93.53 | 92.93 | -0.60 | 2.12x |
| Ours-BN | 93.53 | 93.29 | -0.24 | 2.12x |
| Ours-Group | 93.53 | 93.77 | +0.38 | 2.13x |

Please refer to [benchmarks](benchmarks) for more details.

## Citation
```
@article{fang2023depgraph,
  title={DepGraph: Towards Any Structural Pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
