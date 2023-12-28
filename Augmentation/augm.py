import torch
from tqdm import tqdm
import numpy as np 
import cv2
import os
import random
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt 
from matplotlib.path import Path
import matplotlib.patches as patches
import argparse
import xml.etree.ElementTree as ET
import re

from utils.xml_reader import annot_obj_reader
from utils.distributions import histogram_calc
from utils.Augmenter import Augmenter

parser = argparse.ArgumentParser('Set Augmenter', add_help=False)
parser.add_argument('--image_path', type=str)
parser.add_argument('--annotation_path', type=str)
parser.add_argument('--aug_image_path', type=str)
parser.add_argument('--aug_annotation_path', type=str)
parser.add_argument('--dataset_type', type=str, choices=['HRSC2016','ShipRSImageNet'])

parser.add_argument('--augm_method', type=str, choices=['SSO','ISO','ISC'], default='ISO')
parser.add_argument('--bin_granularity', type=int, default=10)

args = parser.parse_args() 


def main(args):
	image_filenames = np.array(sorted(os.listdir(args.image_path))) # Image filenames in image folder
	annotation_filenames = np.array(sorted(os.listdir(args.annotation_path))) # Annotation filenames in annotations folder

	
	for i in range(len(image_filenames)):
		assert image_filenames[i][:-4] == annotation_filenames[i][:-4]

	if not os.path.exists(args.aug_annotation_path):
		os.mkdir(args.aug_annotation_path)
	if not os.path.exists(args.aug_image_path):
		os.mkdir(args.aug_image_path)

	or_per_cl_distr = histogram_calc(args, annotation_filenames)

	files_n_objs=[]
	files_n_classes=[]
	for ann in annotation_filenames:
		fil_path = os.path.join(args.annotation_path, ann)
		info = annot_obj_reader(fil_path, args.dataset_type)
		if info==[]:
			continue
		else:
			Cls = info[:,0].astype(int)
			n_objs_img = info.shape[0]
			files_n_objs.append(np.array([fil_path, n_objs_img]))
			files_n_classes.append([fil_path, Cls])
	
	files_n_objs = np.array(files_n_objs)
	files_n_objs = np.flip(files_n_objs[files_n_objs[:,1].astype(int).argsort()], axis=0)
	files_n_classes = np.array(files_n_classes, dtype=object)
	augmenter = Augmenter(args)
	augmenter.augment(or_per_cl_distr, files_n_objs, files_n_classes)


if __name__ == '__main__':
	main(args)