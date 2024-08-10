# Vision Transformers Need Zoomer: Efficient Vision Transformers with Local Zooming

This repository contains PyTorch evaluation code, training code and pretrained models for the following projects:

They obtain competitive tradeoffs in terms of speed / precision:

![ZoomViT](ZoomViT.png)



# Model Zoo

We provide baseline DeiT models pretrained on ImageNet 2012.

| name                                  | acc@1 | acc@5 | #params | url                    |
|---------------------------------------|-------|------|---------|------------------------|
| ZoomViT-small-ZF2                     | 83.8  | 96.9 | 22.7M   | [model will be released upon acceptance](https://pan.baidu.com/s/1V-E5rMBkV16L5pEGw-Vs0Q?pwd=heji) |
| ZoomViT-small-ZF0.5                   | 81.8  | 95.7 | 22.7M   | [model will be released upon acceptance](https://pan.baidu.com/s/1Q1BtI0kNA3kdp9t6KLKb4w?pwd=28zg) |

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

## Evaluation
To evaluate a pre-trained DeiT-small-ZF2 on ImageNet val with a single GPU run:
```
python main.py --eval --resume output/best_checkpoint_zf2.pth --data-path /path/to/imagenet
```
This should give
```
* Acc@1 83.816 Acc@5 96.928 loss 0.618
```

For Deit-small-ZF0.5, run:
```
python main.py --eval --resume output/best_checkpoint_zf.5.pth --data-path /path/to/imagenet
```
giving
```
* Acc@1 81.480 Acc@5 95.742 loss 0.730
```
## Train
The training framework for Zoomer will be released later!


# License
This repository is released under the Apache 2.0 license.

# Contributing
We actively welcome your pull requests!  for more info.
