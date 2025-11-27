# Unsupervised learning of spatially varying regularization for diffeomorphic image registration
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2412.17982-b31b1b.svg)](https://arxiv.org/abs/2412.17982)

keywords: Deformable Image Registration, Spatially Varying Regularization

This is a **PyTorch** implementation of my paper:

<a href="https://www.sciencedirect.com/science/article/abs/pii/S1361841525004335">Chen, Junyu, et al. "Unsupervised learning of spatially varying regularization for diffeomorphic image registration." Medical Image Analysis (2025): 103887.</a>

## Updates
11/20/2025 - This work has been accepted for publication in Medical Image Analysis!

## Introduction
***Spatially varying regularization*** accommodates the deformation variations that may be necessary for different anatomical regions during deformable image registration. Historically, optimization-based registration models have harnessed *spatially varying regularization* to address anatomical subtleties. However, most modern deep learning-based models tend to gravitate towards ***spatially invariant regularization***, wherein a homogenous regularization strength is applied across the entire image, potentially disregarding localized variations. In this paper, we propose a hierarchical probabilistic model that integrates a prior distribution on the deformation regularization strength, enabling the ***end-to-end learning of a spatially varying deformation regularizer*** directly from the data.

**I am in the process of consolidating my image registration work into a single package, [MIR](https://github.com/junyuchen245/MIR/tree/main). All future updates will be released there rather than across separate repositories like this one.**

## Registration tasks
* Atlas-to-subject registration on **IXI** dataset (brain MRI) [[code](https://github.com/junyuchen245/Spatially-Varying-Regularization-ImgReg/tree/main/IXI) | [See MIR Package](https://github.com/junyuchen245/MIR/tree/main)]
* Atlas-to-subject registration on **AutoPET** dataset (whole-body CT) [[See MIR Package](https://github.com/junyuchen245/MIR/tree/main)]
* Intra-subject registration on **ACDC** and **M&Ms** datasets (cardiac MRI) [[See MIR Package](https://github.com/junyuchen245/MIR/tree/main)]
* Inter-subject registration on **4DCT** dataset (Lung CT) [[See MIR Package](https://github.com/junyuchen245/MIR/tree/main)]

## Examples of the learned spatially varying regularizer
<img src="https://github.com/junyuchen245/Spatially-Varying-Regularization-ImgReg/blob/main/example_imgs/Qualitative_Results_.jpg" width="800"/>

## Hypernetwork for continuous regularization control
We further incorporated the concept from [HyperMorph](https://www.melba-journal.org/papers/2022:003.html), enabling the learning of a set of regularization hyperparameters for continuous control of spatially varying regularization at the test time.
<img src="https://github.com/junyuchen245/Spatially-Varying-Regularization-ImgReg/blob/main/example_imgs/HyperTMSPR.jpg" width="800"/>

## Citation:
If you find this code is useful in your research, please consider to cite:
    
    @article{chen2025unsupervised,
      title={Unsupervised learning of spatially varying regularization for diffeomorphic image registration},
      author={Chen, Junyu and Wei, Shuwen and Liu, Yihao and Bian, Zhangxing and He, Yufan and Carass, Aaron and Bai, Harrison and Du, Yong},
      journal={Medical Image Analysis},
      pages={103887},
      year={2025},
      publisher={Elsevier}
    }

### <a href="https://junyuchen245.github.io"> About Me</a>
    
