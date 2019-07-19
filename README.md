## Introduction

**PC-DARTS** is a memory-efficient differentiable architecture method based on **DARTS**. It mainly focuses on the large memory cost of the super-net in one-shot NAS method, which means that it can also be combined with other one-shot NAS method e.g. **ENAS**. Different from previous methods that sampling operations, PC-DARTS samples channel of the constructed super-net. For a detailed description of technical details and experimental results, please refer to our paper:

[Partial Channel Connections for Memory-Efficient Differentiable Architecture Search](https://arxiv.org/pdf/1907.05737.pdf)

[Yuhui Xu](http://yuhuixu1993.github.io), [Lingxi Xie](http://lingxixie.com/), [Xiaopeng Zhang](https://sites.google.com/site/zxphistory/), Xin Chen, [Gu-Jun Qi](http://www.eecs.ucf.edu/~gqi/), [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN) and Hongkai Xiong.

**This code is based on the implementation of  [DARTS](https://github.com/quark0/darts).**

## Results
### Results on CIFAR10
Method | Params(M) | Error(%)| Search-Cost
--- | --- | --- | ---
AmoebaNet-B|2.8|2.55|3150
DARTSV1 | 3.3 | 3.00 | 0.4
DARTSV2 | 3.3 | 2.76 | 1.0
SNAS    | 2.8 | 2.85 |1.5
PC-DARTS | 3.6 | **2.57** | **0.1**

Only **0.1 GPU-days** are used for a search on CIFAR-10!
### Results on ImageNet
Method | FLOPs |Top-1 Error(%)|Top-5 Error(%)| Search-Cost
--- | --- | --- | --- | ---
NASNet-A |564|26.0|8.4|1800
AmoebaNet-B|570|24.3|7.6|3150
PNAS     |588 |25.8 |8.1|225
DARTSV2 | 574 | 26.7 | 8.7 | 1.0
SNAS    | 522 | 27.3 | 9.3 |1.5
PC-DARTS | 597 | **24.2** | **7.3** | 3.8

Search a good arcitecture on ImageNet by using the search space of DARTS(**First Time!**).
## Usage

To run our code, you only need one Nvidia 1080ti , and equip it with PyTorch 0.3.1 (python2). (Tesla V100 will be faster).
```
python train_search.py \\
```

#### The evaluation process simply follows that of DARTS.

##### Here is the evaluation on CIFAR10/100:

```
python train_cifar.py \\
       --auxiliary \\
       --cutout \\
```

##### Here is the evaluation on ImageNet (mobile setting):
```
python train_imagenet.py \\
       --tmp_data_dir /path/to/your/data \\
       --save log_path \\
       --auxiliary \\
       --note note_of_this_run
```
## Pretrained models
Coming soon!.

## Notes
- For current mainly file, `python2 with pytorch(3.0.1)` is recommended for the implement on `Nvidia 1080ti`. We also provided codes in the `V100_pytorch1.0` if you want to implement PC-DARTS on `Tesla V100` with `python3+` and `pytorch1.0+`.

- You can search on ImageNet by `model_search_imagenet.py`! The training file for search on ImageNet will be uploaded after it is cleaned or you can generate it according to the train_search file on CIFAR10 and the evluate file on ImageNet. Hyperparameters are reported in our paper!

## Reference

If you use our code in your research, please cite our paper accordingly.
```Latex
@article{xu2019pcdarts,
  title={Partial Channel Connections for Memory-Efficient Differentiable Architecture Search},
  author={Xu, Yuhui and Xie, Lingxi and Zhang, Xiaopeng and Chen, Xin and Qi, Guo-Jun and Tian, Qi and Xiong, Hongkai},
  journal={arXiv preprint arXiv:1907.05737},
  year={2019}
}
