## <b>Detecting Photoshopped Faces by Scripting Photoshop</b> <br>[[Project Page]](http://peterwang512.github.io/FALdetector) [[Paper]](https://arxiv.org/abs/1906.05856)

[Sheng-Yu Wang<sup>1</sup>](https://peterwang512.github.io/),
[Oliver Wang<sup>2</sup>](http://www.oliverwang.info/),
[Andrew Owens<sup>1</sup>](http://andrewowens.com/),
[Richard Zhang<sup>2</sup>](https://richzhang.github.io/),
[Alexei A. Efros<sup>1</sup>](https://people.eecs.berkeley.edu/~efros/). <br>
UC Berkeley<sup>1</sup>, Adobe Research<sup>2</sup>. <br>
In [ICCV, 2019](https://arxiv.org/abs/1906.05856).


<img src='https://peterwang512.github.io/FALdetector/images/teaser.png' align="center" width=900>

<b>9/30/2019 Update</b> The code and model weights have been updated to correspond to the v2 of our paper. Note that the global classifer architecture is changed from resnet-50 to drn-c-26.

<b>1/19/2019 Update</b> Dataset for evaluation is released! The link is [here](https://drive.google.com/file/d/1mzNxCyrUTBF7-lQGPLYT0HuUODvVvtsb/view).

## (0) Disclaimer
Welcome! Computer vision algorithms often work well on some images, but fail on others. Ours is like this too. We believe our work is a significant step forward in detecting and undoing facial warping by image editing tools. However, there are still many hard cases, and this is by no means a solved problem.

This is partly because our algorithm is trained on faces warped by the Face-aware Liquify tool in Photoshop, and will thus work well for these types of images, but not necessarily for others. We call this the "dataset bias" problem. Please see the paper for more details on this issue.

While we trained our models with various data augmentation to be more robust to downstream operations such as resizing, jpeg compression and saturation/brightness changes, there are many other retouches (e.g. airbrushing) that can alter the low-level statistics of the images to make the detection a really hard one.

Please enjoy our results and have fun trying out our models!




## (1) Setup

### Install packages
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### Download model weights
- Run `bash weights/download_weights.sh`


## (2) Run our models
 
### Global classifer
```
python global_classifier.py --input_path examples/modified.jpg --model_path weights/global.pth
```

### Local Detector
```
python local_detector.py --input_path examples/modified.jpg --model_path weights/local.pth --dest_folder out/
```

**Note:** Our models are trained on faces cropped by the dlib CNN face detector. Although in both scripts we included the `--no_crop` option to run the models without face crops, it is used for images with already cropped faces.

## (3) Dataset
A validation set consisting of 500 original and 500 modified images each from Flickr and OpenImage can be downloaded [here](https://drive.google.com/file/d/1mzNxCyrUTBF7-lQGPLYT0HuUODvVvtsb/view). Due to licensing issues, the released validation set is different from the set we evaluate in the paper, and the training set will not be released.

In the zip file, original faces are in the `original` folder, and modified faces are in the `modified` folder. For reference, the `reference` folder contains the same faces in the `modified` folder, but those are before modification (original).

To evaluate the dataset, run:
```
# Download the dataset
cd data
bash download_valset.sh
cd ..
# Run evaluation script. Model weights need to be downloaded.
python eval.py --dataroot data --global_pth weights/global.pth --local_pth weights/local.pth --gpu_id 0
```
The following are the models' performances on the released set:

|Accuracy|  AP |PSNR Increase|
|:------:|:---:|:-----------:|
|   93.9%|98.9%|        +2.66|



## (A) Acknowledgments

This repository borrows partially from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [drn](https://github.com/fyu/drn), and the PyTorch [torchvision models](https://github.com/pytorch/vision/tree/master/torchvision/models) repositories. 

## (B) Citation, Contact

If you find this useful for your research, please consider citing this [bibtex](https://peterwang512.github.io/FALdetector/cite.txt). Please contact Sheng-Yu Wang \<sheng-yu_wang at berkeley dot edu\> with any comments or feedback.
