# Snare drum playing methods classification by using CNN (4-layer CNN and PANNs ResNet38)
## Overview
I used  3 CNN models to classify a single note on the snare drum into 4 techniques(Strike, Rim, Cross Stick and Buzz). In the case of 4-layer CNN, the accuracy is 79.1%.

## Requirement
Python 3.8.3

Pytorch 1.9.1

## Data
[Percussion Dataset](http://www.mattprockup.com/percussion-dataset)

## Model
[4-layer CNN](https://github.com/musikalkemist/pytorchforaudio)

[PANNs ResNet38 (pretrained or not)](https://github.com/qiuqiangkong/audioset_tagging_cnn)

## Run
Please clone this repo and download the [Percusion Dataset](http://www.mattprockup.com/percussion-dataset) and the ResNet38 pretrained model([ResNet38_mAP=0.434.pth](https://zenodo.org/record/3987831#.YdbVTRPP23I)).

The downloaded data looks like:
~~~
-PANNsResNet38_fineturing
-simpleCNN
-data
  └-MDLib2.2
     |-_MACOSX
     └-MDLib2.2
        |-Sorted
        | └-...
        |...
-model
  └-ResNet38_mAP=0.434.pth
~~~
If you want to use 4-layer CNN, run `simpleCNN/train.py`, if you want to use ResNet38, run `PANNsResNet38_finetuning/train4snare.py` with the python command.

You can choose whether or not to pre-train ResNet38 by commenting out line 143 of `PANNsResNet38_finetuning/train4snare.py`.
~~~ 
142:    PRETRAINED_CHECKPOINT_PATH = '../data/model/ResNet38_mAP=0.434.pth'
143:    model.load_from_pretrain(PRETRAINED_CHECKPOINT_PATH)  #If you want to train without pretraining, comment out this line
144:    model = torch.nn.DataParallel(model)
~~~

## Result
|model|epoch|accurancy|
---|---|---
|ResNet38 (pretrained)|30|44.0%|
|ResNet38 (not pretrained)|30|64.8%|
|4-layer CNN|10|79.1%|

## Discussion
For a single percussion instrument, which tends to be seen as monotonous, DNN could detect the differences between the 4 playing methods with about 80% accuracy.

The result is that the accuracy of the deep model is lower. This is different from what is commonly known. It may be that a simple CNN is better suited for this task, or it may be that my code is inadequate. If you notice anything, please message me.

## Reference
- [YouTube channel "Valerio Velardo - The Sound of AI" - Playlist "Pytorch for Audio + Music Processing"](https://youtube.com/playlist?list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm)
- [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://github.com/qiuqiangkong/audioset_tagging_cnn)
