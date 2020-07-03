## Introduction

**PC-DARTS** has been accepted for spotlight presentation at ICLR 2020!

**PC-DARTS** is a memory-efficient differentiable architecture method based on **DARTS**. It mainly focuses on reducing the large memory cost of the super-net in one-shot NAS method, which means that it can also be combined with other one-shot NAS method e.g. **ENAS**. Different from previous methods that sampling operations, PC-DARTS samples channels of the constructed super-net. Interestingly, though we introduced randomness during the search process, the performance of the searched architecture is **better and more stable than DARTS!** For a detailed description of technical details and experimental results, please refer to our paper:

[Partial Channel Connections for Memory-Efficient Differentiable Architecture Search](https://openreview.net/forum?id=BJlS634tPr)

[Yuhui Xu](http://yuhuixu1993.github.io), [Lingxi Xie](http://lingxixie.com/), [Xiaopeng Zhang](https://sites.google.com/site/zxphistory/), Xin Chen, [Guo-Jun Qi](http://www.eecs.ucf.edu/~gqi/), [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN) and Hongkai Xiong.

**This code is based on the implementation of  [DARTS](https://github.com/quark0/darts).**
## Updates
- The implementation of random sampling is also uploaded for your consideration.
- The main file for search on ImageNet has been uploaded `train_search_imagenet.py`.

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
#### Search on CIFAR10

To run our code, you only need one Nvidia 1080ti(11G memory).
```
python train_search.py \\
```
#### Search on ImageNet

Data preparation: 10% and 2.5% images need to be random sampled prior from earch class of trainingset as train and val, respectively. The sampled data is save into `./imagenet_search`.
Note that not to use torch.utils.data.sampler.SubsetRandomSampler for data sampling as imagenet is too large.
```
python train_search_imagenet.py \\
       --tmp_data_dir /path/to/your/sampled/data \\
       --save log_path \\
```
#### The evaluation process simply follows that of DARTS.

##### Here is the evaluation on CIFAR10:

```
python train.py \\
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
- For the codes in the main branch, `python2 with pytorch(3.0.1)` is recommended （running on `Nvidia 1080ti`）. We also provided codes in the `V100_python1.0` if you want to implement PC-DARTS on `Tesla V100` with `python3+` and `pytorch1.0+`.

- You can even run the codes on a GPU with memory only **4G**. PC-DARTS only costs less than 4G memory, if we use the same hyper-parameter settings as DARTS(batch-size=64).

- You can search on ImageNet by `model_search_imagenet.py`! The training file for search on ImageNet will be uploaded after it is cleaned or you can generate it according to the train_search file on CIFAR10 and the evluate file on ImageNet. Hyperparameters are reported in our paper! The search cost 11.5 hours on 8 V100 GPUs(16G each). If you have V100(32G) you can further increase the batch-size.  

- We random sample 10% and 2.5% from each class of training dataset of ImageNet. There are still 1000 classes! Replace `input_search, target_search = next(iter(valid_queue))` with following codes would be much faster:

```
    try:
      input_search, target_search = next(valid_queue_iter)
    except:
      valid_queue_iter = iter(valid_queue)
      input_search, target_search = next(valid_queue_iter)
```

- The main codes of PC-DARTS are in the file `model_search.py`. As descriped in the paper, we use an efficient way to implement the channel sampling. First, a fixed sub-set of the input is selected to be fed into the candidate operations, then the concated output is swaped. Two efficient swap operations are provided: channel-shuffle and channel-shift. For the edge normalization, we define edge parameters(beta in our codes) along with the alpha parameters in the original darts codes. 

- The implementation of random sampling is also provided `model_search_random.py`. It also works while channel-shuffle may have better performance.

- As PC-DARTS is an ultra memory-efficient NAS methods. It has potentials to be implemented on other tasks such as detection and segmentation.

## Related work

[Progressive Differentiable Architecture Search](https://github.com/chenxin061/pdarts)

[Differentiable Architecture Search](https://github.com/quark0/darts)
## Reference

If you use our code in your research, please cite our paper accordingly.
```Latex
@inproceedings{
xu2020pcdarts,
title={{\{}PC{\}}-{\{}DARTS{\}}: Partial Channel Connections for Memory-Efficient Architecture Search},
author={Yuhui Xu and Lingxi Xie and Xiaopeng Zhang and Xin Chen and Guo-Jun Qi and Qi Tian and Hongkai Xiong},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJlS634tPr}
}
