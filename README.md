# Fully-connected Transformer for Multi-source  Image Fusion
This repo is the official code of Fully-connected Transformer for Multi-source  Image Fusion (TPAMI' 2024).

[[Paper]()] [[Project Page](project.md)] [[Data]()] [[Model Zoo]()]

[[UDL]()] [[PanCollection]()] [[HyperSpectralCollection]()] [[MSIF]()] 

## News🔥🔥🔥

* :art: The FC-Former convers multiple multi-source image fusion scenes:
  * Multispectral and hyperspectral image fusion;
  * Remote sensing pansharpening;
  * Visible and infrared image fusion (VIS-IR); 
  * Digital photographic image fusion: Multi-focus image fusion (MFF) and multi-exposure image fusion (MEF). 

* 🎁 We will release a new version of [UDL](), [PanCollection](). Furthermore, we also release repositories of [HyperSpectralCollection]() and [MSIF]().

* :eight_pointed_black_star: **The FC-Former employs multilinear algebra to unify and generalize self-attention mechanisms.** One of the generalized self-attention mechanisms is fully-connected self-attention (FCSA).

* 👍 The FC-Former achieves the state-of-the-art performance with less parameter number and have a powerful potential in feature representation, which is a win-win situation.



## Quick Start 🤗
1. Install base training/inference repo with multiple Pytorch framework support. You can select one of backends: accelerate, lightning, transformers, and mmcv1.
```
pip install udl_vis --upgrade
```
2. Install the [Task] repo by populating one of the following repos: ``pancollection``, ``mhif`` ``msif``.

``` 
pip install [Task] --upgrade
```

3. Then in this repository:
```
pip install -e .
```



## Train
Here, we take Huggingface ``accelerate`` as the backend, thus
```
sh  run_accelerate_ddp_pansharpening.sh
```
The script will call ``python_scripts/accelerate_pansharpening.py`` to run the FC-Former. More information you can see ``model.yaml``


## Inference 
Inference the FC-Former to obtain final results, you only need to update the `model.yaml` as follows:
```yaml
  eval : true # change false to true
  workflow:
    - ["test", 1]
    # - ["train", 10] # comment it
```




## Test Your Metrics
Finally, we provide the correspoinding Matlab toolboxes to test the metrics on those tasks. Please check them in our repo.

* MHIF
  * [HyperSpectralToolbox, comming soon](https://github.com/XiaoXiao-Woo/HyperSpectralToolbox)
  * `run_hisr.m`
  
* Pansharpening
  * [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox)
  
* VIS-IR, MEF, MFF
  * [Image Fusion Toolbox, comming soon]()



## Experiments
In this part, we conduct experiments in the following cases. All settings can be changed in `dataset.yaml`.
* MHIF: CAVE and Harvard. 
  * See our repo [MHIF]() for more details;
* VIS-IR image fusion: TNO and RoadScene datasets. 
  * See our repo [MSIF]() for more details;
* Pansharpening: WorldView-3, GaoFen-2, QuickBird datasets.
  * See our repo [PanCollection]() for more details;
* Digital photographic image fusion: MFF-WHU, MEF-Lytro,MEF-SLICE, MEFB. 
  * See our repo [MSIF]() for more details;




## Citation
Please cite this project if you use datasets or the toolbox in your research.
```bibtex
@article{FCForrmer,
  title={Fully-connected Transformer for Multi-source  Image Fusion},
  author={Xiao Wu, Zi-Han Cao, Ting-Zhu Huang, Liang-Jian Deng, Jocelyn Chanussot, and Gemine Vivone}
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}

@inproceedings{Wu_2021_ICCV,
    author    = {Wu, Xiao and Huang, Ting-Zhu and Deng, Liang-Jian and Zhang, Tian-Jing},
    title     = {Dynamic Cross Feature Fusion for Remote Sensing Pansharpening},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14687-14696}
}

@article{zhong2024ssdiff,
  title={SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening},
  author={Zhong, Yu and Wu, Xiao and Deng, Liang-Jian and Cao, Zihan},
  journal={arXiv preprint arXiv:2404.11537},
  year={2024}
}

@ARTICLE{duancvpr2024,
title={Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening},
author={Yule Duan, Xiao Wu, Haoyu Deng, Liang-Jian Deng*},
journal={IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
year={2024}
}

@ARTICLE{dengijcai2023,
title={Bidirectional Dilation Transformer for Multispectral and Hyperspectral Image Fusion},
author={Shang-Qi Deng, Liang-Jian Deng*, Xiao Wu, Ran Ran, Rui Wen},
journal={International Joint Conference on Artificial Intelligence (IJCAI)},
year={2023}
}

@misc{PanCollection,
    author = {Xiao Wu, Liang-Jian Deng and Ran Ran},
    title = {"PanCollection" for Remote Sensing Pansharpening},
    url = {https://github.com/XiaoXiao-Woo/PanCollection/},
    year = {2022},
}
```


## License
This project is open sourced under GNU General Public License v3.0.
