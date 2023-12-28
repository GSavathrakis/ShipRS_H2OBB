import argparse
import os
import numpy as np
from utils.xml_reader import annot_reader_and_remover


parser = argparse.ArgumentParser(
	description='Per orientation precision calculation')
#parser.add_argument('config', help='test config file path')
parser.add_argument('--dataset', type=str, choices=['HRSC2016', 'ShipRSImageNet'])
#parser.add_argument('--test_image_path', type=str, help='Path to test image directory')
parser.add_argument('--test_annotation_path', type=str, help='Path to test annotations directory')
parser.add_argument('--orientation_annotations_dir', type=str, help='folder with per orientation annotations')
parser.add_argument('--bin_granularity', default=10, help='Discretization of rotation space')
#parser.add_argument('--detector_script', type=str, help='Path to openmmlab detector')
	
args = parser.parse_args()
	


def main(args):
	if not os.path.exists(args.orientation_annotations_dir):
		os.mkdir(args.orientation_annotations_dir)
		dataset_dir = os.path.join(args.orientation_annotations_dir, args.dataset)
		os.mkdir(dataset_dir)
	#print(np.linspace(0,170, 180//args.bin_granularity))
	annotation_filenames = np.array(sorted(os.listdir(args.test_annotation_path)))
	for rot in np.linspace(0, 170, 180//args.bin_granularity):
		print(f'finding files with objects at angle {int(rot)}')
		rot_dir = f'rotation_{int(rot)}'
		rot_dataset_dir = os.path.join(dataset_dir, rot_dir)
		if not (os.path.exists(rot_dataset_dir)):
			os.mkdir(rot_dataset_dir)
		for ann in annotation_filenames:
			#fil_path = os.path.join(args.test_annotation_path, ann)
			objs_info = annot_reader_and_remover(args.dataset, args.test_annotation_path, rot_dataset_dir, ann, int(rot), args.bin_granularity)
			
		

if __name__ == '__main__':
	main(args)