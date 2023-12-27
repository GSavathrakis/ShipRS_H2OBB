# Hbb_to_Obb
The official implementation of the paper [An Automated Method for the Creation of Oriented Bounding Boxes in Remote Sensing Ship Detection Datasets](https://openaccess.thecvf.com/content/WACV2024W/MaCVi/html/Savathrakis_An_Automated_Method_for_the_Creation_of_Oriented_Bounding_Boxes_WACVW_2024_paper.html)\
Given images with objects with Horizontal Bounding Box annotations, the proposed method can create reliable Oriented Bounding Boxes, with a pipeline consisting of the Segment Anything Model (SAM), a morphological closing and a contour detection operation.\
\
\
![alt text](https://github.com/GSavathrakis/hbb_to_obb/blob/main/Figures/model_arch.png?raw=true)
\
Additionally, this method can be used to create augmented versions of the dataset in a manner that resolves the objects' orientation imbalance.
# Installation
Firstly, an anaconda environment needs to be set with a python>=3.9
```
conda create -n "env name" python="3.9 or above"
conda activate "env name"
```
Then install pytorch, torchvision and the segment anything model
```
conda install pytorch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git
```
# Usage
Initially, clone the repository.
```
git clone https://github.com/GSavathrakis/hbb_to_obb.git
cd hbb_to_obb
```
Download the segment-anything model checkpoint from the [official segment-anything model repo](https://github.com/facebookresearch/segment-anything)
## Obb generation
To use the method for the creation of OBBs from HBBs , run
```
python OBB_generation/generate.py --dataset "dataset name" --image_path "The path to the images directory" --annotation_path "The path to the annotations directory" --sam_checkpoint_path "The path to where the segment-anything checkpoint is stored" --new_annotations_dir "The path where the newly created OBB annotations will be saved" --gen_mode
```
## Data augmentation
For the creation of augmented datasets with uniform object orientation distribution, run
```
python Augmentation/augm.py --image_path "The path to the images directory" --annotation_path "The path to the annotations directory" --aug_image_path "The path to the directory where the augmented images will be saved" --aug_annotation_path "The path to the directory where the annotations of the augmented images will be saved" --dataset_type "dataset name" --augm_method "The augmentation method to be used"
```
# Acknowledgements
<details>
  <summary>Expand</summary>
    * [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
    * [https://github.com/open-mmlab/mmrotate](https://github.com/open-mmlab/mmrotate)
</details>
