# Fine-Grained-FSCIL
Code for Optimising Few-Shot Class-Incremental Learning for Fine-Grained Visual Recognition
# Dataset
The Aircraft, Stanford Cars and Stanford dogs can download from [here](https://github.com/zichengpan/PFR). For CUB200, mini-ImageNet and CIFAR100, please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.
# Training Scripts
CUB200
```
$python train.py -project lgf -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.005 -lr_new 0.000005 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -mlp -moco_t 0.07 -size_crops 224 96 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 0 -constrained_cropping
```
Aircraft
```
$python train.py -project lgf -dataset aircraft100 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.05 -batch_size_base 16 -alpha 0.8 -lr_new 0.000005 -decay 0.0005 -epochs_base 120 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -mlp -moco_t 0.07 -size_crops 84 21 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 0 -constrained_cropping 
```
Stanford Cars
```
$python train.py -project lgf -dataset car100 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.05 -batch_size_base 16 -alpha 0.8  -beta 0.8 -lr_new 0.000005 -decay 0.0005 -epochs_base 200 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -mlp -moco_t 0.07 -size_crops 84 21 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 0 -constrained_cropping
```
Stanford Dogs
```
$python train.py -project lgf -dataset StanfordDog -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.05 -batch_size_base 16 -alpha 0.8 -lr_new 0.000005 -decay 0.0005 -epochs_base 150 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -mlp -moco_t 0.07 -size_crops 84 21 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 0 -constrained_cropping
```
mini-ImageNet
```
$python train.py -project lgf -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -lr_new 0.001 -decay 0.0005 -alpha 0.8 -epochs_base 250 -schedule Cosine -gpu 0 -temperature 16 -moco_dim 32 -mlp -moco_t 0.07 -size_crops 84 50 -min_scale_crops 0.9 0.2 -max_scale_crops 1.0 0.7 -num_crops 2 0 -constrained_cropping
```
CIFAR100
```
$python train.py -project lgf -dataset cifar100 -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -alpha 0.8 -lr_new 0.001 -decay 0.0005 -epochs_base 120 -schedule Cosine -gpu 0 -temperature 16 -moco_dim 32 -mlp -moco_t 0.07 -size_crops 32 18 -min_scale_crops 0.9 0.2 -max_scale_crops 1.0 0.7 -num_crops 2 0 -constrained_cropping
```
# Detailed Performance
<img width="920" height="829" alt="image" src="https://github.com/user-attachments/assets/39bca8e4-fa58-4f0c-a424-eb67ff7bc0c7" />

<img width="922" height="310" alt="image" src="https://github.com/user-attachments/assets/41579b16-9e40-40b6-b66f-f5f8a42c5339" />

<img width="911" height="319" alt="image" src="https://github.com/user-attachments/assets/8fcf8565-7b2b-4670-970c-8573a67a86b6" />

<img width="914" height="319" alt="image" src="https://github.com/user-attachments/assets/d87fae37-afd3-4d74-a643-d5466f6859b0" />


