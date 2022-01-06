# Snare drum playing methods classification by using CNN (SimpleCNN and PANNs ResNet38)
## Over View
I used  3 CNN models to classify a single note on the snare drum into four techniques(Strike, Rim, Cross Stick and Buzz). In the case of Simple CNN, the accuracy is 79.1%.

## Requirement
Python 3.8.3

Pytorch 1.9.1

## Data
[Percussion Dataset](http://www.mattprockup.com/percussion-dataset)

## Model
[Simple CNN](https://github.com/musikalkemist/pytorchforaudio)

[ResNet38 (pretrained or not)](https://github.com/qiuqiangkong/audioset_tagging_cnn)

## Run
Please download the Percusion Dataset and the ResNet38 pretrained model(ResNet38_mAP=0.434.pth) and the downloaded data looks like:
'''
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
'''
If you want to use Simple CNN, run 'simpleCNN/train4snare.py', if you want to use ResNet38, run 'snaredrum-playing-methods-classification/snaredrum-playing-methods-classification/PANNsResNet38_finetuning/train.py' with the python command.
You can choose whether or not to pre-train ResNet38 by commenting out line 142 of 'snaredrum-playing-methods-classification/PANNsResNet38_finetuning/train4snare.py'.
'''
141:  
142:    PRETRAINED_CHECKPOINT_PATH = '../data/model/ResNet38_mAP=0.434.pth' #If you want to train without pretraining, comment out this line
143:    model.load_from_pretrain(PRETRAINED_CHECKPOINT_PATH)
'''

