# Hbb_to_Obb
The official implementation of the paper [An Automated Method for the Creation of Oriented Bounding Boxes in Remote Sensing Ship Detection Datasets](https://openaccess.thecvf.com/content/WACV2024W/MaCVi/html/Savathrakis_An_Automated_Method_for_the_Creation_of_Oriented_Bounding_Boxes_WACVW_2024_paper.html)\
Given images with objects with Horizontal Bounding Box annotations, the proposed method can create reliable Oriented Bounding Boxes, with a pipeline consisting of the Segment Anything Model (SAM), a morphological closing and a contour detection operation.\
\
\
![alt text](https://github.com/GSavathrakis/hbb_to_obb/blob/main/Figures/model_arch.png?raw=true)
\
Additionally, this method can be used to create augmented versions of the dataset in a manner that resolves the objects' orientation imbalance.
# Installation
