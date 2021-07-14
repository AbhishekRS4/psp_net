# PSPNet implementation on Cityscapes dataset

## Notes
* Implementation of PSPNet with ResNet-50
* The original implementation uses ResNet-101 for cityscapes dataset
* The image dimension used to train the model is 1024x512
* 15 custom classes used

## Main idea
* Apply pyramid pooling to feature maps of output stride 8 of input size and concatenate the output of pyramid pooling block to its input. Perform bilinear upsampling by a factor of 8

## Intructions to run
* To list training args use
```
python3 psp_net_train.py --help
```
* To list inference args use
```
python3 psp_net_infer.py --help
```

## Visualization of results - Youtube video
[![IMAGE ALT TEXT](http://img.youtube.com/vi/DPIeSIGCvBs/0.jpg)](https://www.youtube.com/watch?v=DPIeSIGCvBs "psp-net cityscapes")

## Reference
* [ResNet-50](https://arxiv.org/abs/1512.03385)
* [PSPNet](https://arxiv.org/pdf/1612.01105.pdf)
* [PSPNet Project](https://hszhao.github.io/projects/pspnet/index.html)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
