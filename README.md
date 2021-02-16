# openpifpaf_wholebody
This is an extension to [Pifpaf](https://github.com/vita-epfl/openpifpaf) to detect body, foot, face and hand keypoints, which sum up to 133 keypoints per person. The annotations for these keypoints are taken from the [COCO WholeBody dataset](https://github.com/jin-s13/COCO-WholeBody). <br/> Example outputs and skeleton:
![Soccer players with superimposed predictions](/docs/0001soccer.jpeg.predictions.png)

[Image](https://de.wikipedia.org/wiki/Kamil_Vacek#/media/Datei:Kamil_Vacek_20200627.jpg) licensed under [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
The superimposed poses were predicted with:
```
python -m openpifpaf.predict 0001soccer.jpeg --checkpoint=shufflenetv2k30-wholebody --show --line-width=2 --decoder=cifcaf:0
```

![Skeleton](/docs/skeleton_wholebody.png)

## Install via pip
You can use pip to install openpifpaf wholebody. From the openpifpaf_wholebody folder run:
```
pip3 install openpifpaf_wholebody
```
This will also automatically install openpifpaf, if it is not already installed.

## Getting started
Start in an empty folder. Create an environment, clone the pifpaf dev branch and install pifpaf:
```
git clone --single-branch --branch dev https://github.com/vita-epfl/openpifpaf.git
python3 -m venv ./pifpaf_env
source ./pifpaf_env/bin/activate 
cd openpifpaf
pip3 install --editable '.[dev,train,test]'
```
Finally, clone the pifpaf foot extension and soft link the ms coco annotations and images, and download the wholebody annotations:
```
git clone https://github.com/vita-epfl/openpifpaf_wholebody.git
mkdir ./openpifpaf_wholebody/data-mscoco
ln -s /<PathToYourMSCOCO>/data-mscoco/images ./openpifpaf_wholebody/data-mscoco/images
ln -s /<PathToYourMSCOCO>/data-mscoco/annotations ./openpifpaf_wholebody/data-mscoco/annotations
wget https://github.com/DuncanZauss/openpifpaf_assets/releases/download/v0.1.0/person_keypoints_train2017_wholebody_pifpaf_style.json -O ./openpifpaf_wholebody/data-mscoco/annotations
wget https://github.com/DuncanZauss/openpifpaf_assets/releases/download/v0.1.0/person_keypoints_val2017_wholebody_pifpaf_style.json -O ./openpifpaf_wholebody/data-mscoco/annotations
```
Note: The pifpaf style annotation files were create with [Get_annotations_from_coco_wholebody.py](/Helper_scripts/Get_annotations_from_coco_wholebody.py). If you want to create your own annotation files from coco wholebody, you need to download the original files from the [Coco Whole body page](https://github.com/jin-s13/COCO-WholeBody#download) and then create the pifpaf readable json files with [Get_annotations_from_coco_wholebody.py](/Helper_scripts/Get_annotations_from_coco_wholebody.py). This can be useful if you for example only want to use a subset of images for training.

## Show poses
Visualize the human poses with 133 keypoints.
```
python -m openpifpaf_wholebody.src.constants
```

## Predict
Use the pretrained model to perform a prediction:<br/>
`python -m openpifpaf.predict ./openpifpaf_wholebody/docs/0001basketball.jpeg --checkpoint=shufflenetv2k30-wholebody --show --line-width=1 --decoder=cifcaf:0`
<br/> <br/> Note that `--decoder=cifcaf:0` has to be used to use the first decoder. As the pretrained model was trained on two datasets to achieve better performance, it has two decoders.

## Train
If you don't want to use the pre-trained model, you can train a model from scratch.
To train you first need to download the wholebody into your MS COCO dataset folder:
```
wget https://github.com/DuncanZauss/openpifpaf_assets/releases/download/v0.1.0/person_keypoints_train2017_wholebody_pifpaf_style.json -O /<PathToYourMSCOCO>/data-mscoco/annotations
wget https://github.com/DuncanZauss/openpifpaf_assets/releases/download/v0.1.0/person_keypoints_val2017_wholebody_pifpaf_style.json -O /<PathToYourMSCOCO>/data-mscoco/annotations
```
Note: The pifpaf style annotation files were create with [Get_annotations_from_coco_wholebody.py](/openpifpaf_wholebody/Helper_scripts/Get_annotations_from_coco_wholebody.py). If you want to create your own annotation files from coco wholebody, you need to download the original files from the [Coco Whole body page](https://github.com/jin-s13/COCO-WholeBody#download) and then create the pifpaf readable json files with [Get_annotations_from_coco_wholebody.py](/openpifpaf_wholebody/Helper_scripts/Get_annotations_from_coco_wholebody.py). This can be useful if you for example only want to use a subset of images for training.

Finally you can train the model (Note: This can take several days, even on the good GPUs):<br/>
`time CUDA_VISIBLE_DEVICES=0 python3 -m openpifpaf.train --lr=0.0003 --momentum=0.95 --b-scale=3.0 --epochs=150 --lr-decay 130 140 --lr-decay-epochs=10 --batch-size=16 --weight-decay=1e-5 --dataset=wholebodykp --wholebodykp-upsample=2 --basenet=shufflenetv2k16 --loader-workers=16 --wholebodykp-train-annotations=<PathToYourMSCOCO>/data-mscoco/annotations/person_keypoints_train2017_wholebody_pifpaf_style.json --wholebodykp-val-annotations=<PathToYourMSCOCO>/data-mscoco/annotations/person_keypoints_val2017_wholebody_pifpaf_style.json --wholebodykp-train-image-dir=<COCO_train_image_dir> --wholebodykp-val-image-dir=<COCO_val_image_dir>`

## Evaluation
To evaluate your network you can use the following command:<br/>
`time CUDA_VISIBLE_DEVICES=0 python3 -m openpifpaf.eval --checkpoint=shufflenetv2k16-wholebody --force-complete-pose --seed-threshold=0.2 --loader-workers=16 --wholebodykp-val-annotations=<PathToYourMSCOCO>/data-mscoco/annotations/person_keypoints_val2017_wholebody_pifpaf_style.json --wholebodykp-val-image-dir=<COCO_val_image_dir>`

## Using only a subset of keypoints
If you only want to train on a subset of keypoints, e.g. if you do not need the facial keypoints and only want to train on the body, foot and hand keypoints, it should be fairly easy to just train on this subset. You will need to:
- Download the original annotation files from the [Coco Whole body page](https://github.com/jin-s13/COCO-WholeBody#download). Create a new annotations file with [Get_annotations_from_coco_wholebody.py](/Helper_scripts/Get_annotations_from_coco_wholebody.py). Set `ann_types`to the keypoints that you would like to use and create the train and val json file. You can use [Visualize_annotations.py](/Helper_scripts/Visualize_annotations.py.py) to verify that the json file was created correctly.
- In the [constants.py](/openpifpaf_wholebody/constants.py) file comment out all the parts of the skeleton, pose, HFLIP, SIGMA and keypoint names that you do not need. All these constants are already split up in the body parts. The numbering of the joints may now be different (e.g. when you discard the face kpts, but keep the hand kpts), so you need to adjust the numbers in the skeleton definitions to be consisten with the new numbering of the joints.
- That's it! You can train the model with a subset of keypoints.

## Further informations
For more information refer to the [Pifpaf Dev Guide](https://vita-epfl.github.io/openpifpaf/dev/intro.html).
