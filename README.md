# Evaluation of CLIP image features extractors
This repository contains experiments and results of comparison of image features extractors generated by classical training on ImageNet and CLIP[[paper](https://arxiv.org/abs/2103.00020)][[repo](https://github.com/openai/CLIP)][[blog](https://openai.com/blog/clip/)] training procedure on a specific [Fruits](link.to.dataset) dataset.

### **Zero-shot prediсtions**
The procedure described in the CLIP [paper](https://arxiv.org/abs/2103.00020) allows to make predictions on a new image dataset with any set of labels without training.
Caption format: **Predicted (True)**
<p align="center"><img src="pics/zero-shot-visualization.jpg" width="700" /></p>


## Experiments accomplished
We compared features extractors with different architectures, training procedures and image upsampling techniques. If an image upsampling technique is not mentioned, then bicubic interpolation is used. We performed the following two main sections of experiments:
1. Linear probing and fine-tuning of CLIP with ResNet and ViT backbones and ImageNet-pretrained ResNet and EfficientNet
2. Zero-shot and K-shot classification of CLIP with ViT and ResNet backbones

We also compared 2 image upsampling options:
 - Bucubic interpolations
 - SRGAN upsampling [[weights](https://drive.google.com/drive/folders/1-_0wNvmjFnISr_lN520DfqyqN3uydgFC?usp=sharing)]

We did it on the following training setups: linear probing and contrastive fine-tuning of CLIP with ResNet and ViT backbones.

Main plots can be found in the [results](#results) section. Full experiments descriptions can be found in the ```supplementary/report.pdf```

## Repository structure
- ```notebooks/``` — contains experiments in form of jupyter notebooks \
    ```├── few_shot_learning.ipynb``` — k-shot learning procedure\
    ```├── image_upsampling.ipynb``` — two ways to upsample images with subsequent saving\
    ```├── prompts_validation.ipynb``` — finding the best prompt for given dataset\
    ```├── train_ImageNet_models.ipynb``` — fine-tuning of models pretrained on ImageNet in different settings\
    ```└── train_CLIP.ipynb``` — fine-tuning CLIP models in different settings
- ```data_prepare/``` — Dataset upsampling auxilary source code
- ```src/``` — training related auxilary source code
- ```pics/``` — pictures for the [results](#results) part
- ```supplementary/``` — contains report and presentation in ```.pdf``` format



## Results

### K-shot training
Pretained CLIP model with **ResNet-101** backbone + new fully-connected layer which is trained only on *k* examples of each class.
<p align="center"><img src="pics/few-shot.png" width="450" /></p>

### Fine-tuning
Fune-tuning of visual parts of CLIP models with linear classifier on top with frozen/trainable backbones

<p align="center"><img src="pics/CLIP_lin_train.png" width="400"/> <img src="pics/CLIP_lin_val.png" width="400"/></p> 


### Linear probing
Pretained model with new fully-connected layer on top which is trained on a training part of the target dataset.
<p align="center"><img src="pics/fine-tuning-plot.jpg" width="450" /></p>

### Upsampling
<p align="center"><img src="pics/interpolations.jpg" width="450" /></p>

