# Fine-Grained-FSCIL
Code for Optimising Few-Shot Class-Incremental Learning for Fine-Grained Visual Recognition
# Dataset
The Aircraft, Stanford Cars and Stanford dogs can download from [here](https://github.com/zichengpan/PFR). For CUB200, mini-ImageNet and CIFAR100, please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.
# Training Scripts
CUB200
```
$python train.py -project lgf -dataset cub200 -alpha 1.0 -beta 0.4 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.005 -lr_new 0.000005 -decay 0.0005 -epochs_base 100 -schedule Milestone  -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -mlp -moco_t 0.07 -size_crops 224 96 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 0 -constrained_cropping
```
Aircraft
```
$python train.py -project lgf -dataset aircraft100 -alpha 0.2 -beta 0.4 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.05 -batch_size_base 16 -lr_new 0.000005 -decay 0.0005 -epochs_base 120 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -mlp -moco_t 0.07 -size_crops 84 21 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 0 -constrained_cropping 
```
Stanford Cars
```
$python train.py -project lgf -dataset car100 -alpha 0.6 -beta 0.4 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.05 -batch_size_base 16 -lr_new 0.000005 -decay 0.0005 -epochs_base 200 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -mlp -moco_t 0.07 -size_crops 84 21 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 0 -constrained_cropping
```
Stanford Dogs
```
$python train.py -project lgf -dataset StanfordDog -alpha 0.8 -beta 0.6 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.05 -batch_size_base 16 -lr_new 0.000005 -decay 0.0005 -epochs_base 150 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -mlp -moco_t 0.07 -size_crops 84 21 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 0 -constrained_cropping
```
mini-ImageNet
```
$python train.py -project lgf -dataset mini_imagenet -alpha 1.0 -beta 0.6 -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -lr_new 0.001 -decay 0.0005 -epochs_base 250 -schedule Cosine -gpu 0 -temperature 16 -moco_dim 32 -mlp -moco_t 0.07 -size_crops 84 50 -min_scale_crops 0.9 0.2 -max_scale_crops 1.0 0.7 -num_crops 2 0 -constrained_cropping
```
CIFAR100
```
$python train.py -project lgf -dataset cifar100 -alpha 0.4 -beta 0.4 -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -lr_new 0.001 -decay 0.0005 -epochs_base 120 -schedule Cosine -gpu 0 -temperature 16 -moco_dim 32 -mlp -moco_t 0.07 -size_crops 32 18 -min_scale_crops 0.9 0.2 -max_scale_crops 1.0 0.7 -num_crops 2 0 -constrained_cropping
```
