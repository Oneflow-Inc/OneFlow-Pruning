```shell
# This directory contains code for pruning neural networks using OneFlow.
# The directory contains two subdirectories: pruner and utils.
# The pruner subdirectory contains code for different pruning algorithms and a scheduler.
# The utils subdirectory contains code for counting the number of operations in a neural network.
# The files in the root directory are helper files for the code in the subdirectories.
# The __init__.py files are empty files that indicate that the directories are Python packages.
# The importance.py file contains code for computing the importance of each layer in a neural network.
# The ops.py file contains code for pruning convolutional and linear layers.
# The dependency.py file contains code for checking the dependencies of the OneFlow pruning module.
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

