import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys
sys.path.append("/workspace/Augmentation")
from utils.distributions import histogram_calc

parser = argparse.ArgumentParser('Select annotations folder', add_help=False)
parser.add_argument('--annotation_path', type=str)
parser.add_argument('--dataset_type', type=str, choices=['DOTA_v1.5','HRSC2016','ShipRSImageNet'])
parser.add_argument('--bin_granularity', type=int, default=10)
parser.add_argument('--image_save_name', type=str)
args = parser.parse_args() 

def main(args):
	annot_files = sorted(np.array(os.listdir(args.annotation_path)))
	hist = histogram_calc(args, annot_files)
	#print(hist.sum(axis=0))
	
	plt.figure(figsize=(7,6))
	plt.vlines(np.linspace(0, 180-args.bin_granularity, 180//args.bin_granularity), 0, hist.sum(axis=0), linewidth=20)
	plt.xticks(np.linspace(0, 180-args.bin_granularity, 180//args.bin_granularity), fontsize=10)
	plt.yticks()
	plt.xlabel('Orientations (deg)', fontsize=15)
	plt.ylabel('# Objects', fontsize=15)
	plt.savefig(args.image_save_name)
	plt.close()
	

if __name__ == '__main__':
	main(args)