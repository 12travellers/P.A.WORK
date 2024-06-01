### Requirements

```
pip install numpy torch torchvision tqdm tensorboard
```

### Usage

##### Dataset

The original picture data should be put in *data_folder* in the form like:

```
P.A.WORK\
|		- data\
|			- train\
|					- 0\
|						- ***.png
|						- ***.png
|						- ......
|					- 1\
|						- ***.png
|						- ......
|			- test\
|					- 0\
|						- ***.png
|						- ......
|					- 1\
|						- ***.png
|						- ......
```

, similar to CIFAR-10.

##### Code

```
python train.py --data_pth=[data folder] --best_ckpt_path=[checkpoint save path]
```

*train.py* finetunes a ResNet model from torchvision's pretrained weights and save it to *ckpt*. Notice that to use which ResNet model as the pretrained one is decided through the name of your saved checkpoint's name. See more training configurations in the code.

```
python fool.py --data_pth=[data folder] --best_ckpt_path=[proxy checkpoint save path] --output=[path storing the faked data] 
```

*fool.py* takes the proxy model generates adversarial sample through training a Unet model on each picture in the data folder. Notice that to use which ResNet structure is decided through the name of your saved checkpoint's name. See more configurations in the code.

```
python train.py --data_pth=[data folder] --best_ckpt_path=[checkpoint save path]
```

*train.py* runs the model weights and run through the data to get output. It will also create a .txt file in *output* folder recording the details of the evaluation. Notice that to use which ResNet structure is decided through the name of your saved checkpoint's name. 