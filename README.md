# Object Detection training and testing scripts
*by Cyrille C. Cervantes*

This repository contains the training, testing as well as the demonstration scripts for the object detection of three classes, namely,
- **Summit Drinking Water 500ml**
- **Coca-Cola 330ml** 
- **Del Monte 100% Pineapple Juice 240ml**

The scripts were modified version of the [Python Vision Object Detection](https://github.com/pytorch/vision/tree/main/references/detection). The codes were modified as to conform to the custom dataset acquired by the author. With that, the models that can be used for training are limited to Pytorch models. To name, they are [Faster R-CNN](https://arxiv.org/abs/1506.01497), [FCOS](https://arxiv.org/abs/1904.01355), [RetinaNet](https://arxiv.org/abs/1708.02002), [SSD](https://arxiv.org/abs/1512.02325) and [SSDlite](https://arxiv.org/abs/1801.04381). *Mask R-CNN is not supported*.


### Install
To start, you may install the required python packages for the scripts by running the following command
```
pip install -r requirements.txt
```
## Training
To train, simply run the ```train.py``` script. The training specs as well other confiurations can be specified using the command line arguments. Here is an example command.
```
python train.py --model fasterrcnn_mobilenet_v3_large_fpn\
  --batch-size 8 \
  --workers 2 \
  --epochs 26 \
  --output-dir checkpoints/fasterrcnn_mobilenet_v3_large_fpn \
```
Here are some of the arguments that can be configured are listed below. For the full list see train.py.
| arg | default | description|
|:---:|:---:|:---:|
|--model|fasterrcnn_mobilenet_v3_large_fpn| Model name. Please look up [here](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection) for supported models|
|--device|cuda|device (Use cuda or cpu Default: cuda)|
|-b, --batch-size|4|images per gpu, the total batch size is $NGPU x batch_size|
|--epochs|26|number of total epochs to run|
|-j, --workers|4|number of data loading workers (default: 4)|
|--opt|sgd|optimizer|
|-lr|0.02|initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu|
|--momentum|0.9|momentum|
|--wd,--weight-decay|1e-4|weight decay (default: 1e-4)|
|--norm-weight-decay|None|weight decay for Normalization layers (default: None, same value as --wd)|
|--lr-scheduler|multisteplr|name of lr scheduler (default: multisteplr)|
|--lr-step-size|8|decrease lr every step-size epochs (multisteplr scheduler only)|
|--lr-steps|[16, 22]|decrease lr every step-size epochs (multisteplr scheduler only)|
|--lr-gamma|0.1|decrease lr by a factor of lr-gamma (multisteplr scheduler only)|
|--output-dir|checkpoints|path to save outputs|
|--resume||path of checkpoint|
|--start_epoch|0|start epoch|
|--aspect-ratio-group-factor|3||
|--data-augmentation|hflip|data augmentation policy (default: hflip)|

## Testing / Evaluation
To test, just simply run ```test.py```. Same with the train.py, some settings can be configured through command line arguments.
- When ```--use-pretrained``` is used, the script will download the pretrained weights of the specified ```--model```.
- When --use-pretrained is not used, the script will test the weights acquired from train.py and thus it is required that --model is the same for both the train.py and test.py.

Here are some examples.
```
python test.py
```
```
python test.py --model fasterrcnn_mobilenet_v3_large_fpn --use_pretrained
```
## Demo
*scripts coming soon*
For now, here is a video demonstration of the real-time object detection.

https://user-images.githubusercontent.com/39228574/166243786-824e219d-0694-4c24-920d-373414aea20e.mp4


