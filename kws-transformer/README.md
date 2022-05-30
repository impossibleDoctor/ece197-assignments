# Keyword Spotting using Transformer
*by Cyrille C. Cervantes*

This repository contains the training, testing and demo scripts for keyword spotting using a Transformer.

### Install
Before you start, you may install the required python packages for the scripts by running the following command
```
pip install -r requirements.txt
```
## Training
To train, simply run the ```train.py``` script.
:warning: **It is recommended not to change parameters for MelSpectrogram as doing so may give inconsistent input dimensions to the transformer.**
```
python train.py
```

## Demo
To run a demo, you may use the ```kws-infer.py```. It will run an inference on a single audio file or an input from the device's microphone.
Simple test:
```
python3 kws-infer.py --wav-file <path-to-wav-file>  
```
To use your microphone as an input with GUI interface:
```
python3 kws-infer.py --gui
```

Here is a video demonstration of the real-time keyword spotting.

https://user-images.githubusercontent.com/39228574/170971546-25bdbb34-dfba-456e-93ea-acf439f0061f.mp4


