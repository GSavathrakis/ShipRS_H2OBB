import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys
sys.path.append("/workspace/Augmentation")
from utils.distributions import histogram_calc

#cl_n_angs = np.loadtxt('/workspace/classes_n_angles_spl_1pt_06.txt', delimiter=' ')
#class_names = #np.loadtxt('Path to csv file with class names', dtype=str)
#classes = cl_n_angs[:,0]
#angles = cl_n_angs[:,1]
#angles_corr = angles[np.where(angles<180)]

parser = argparse.ArgumentParser('Select annotations folder', add_help=False)
parser.add_argument('--annotation_path', type=str)
parser.add_argument('--dataset_type', type=str, choices=['DOTA_v1.5','HRSC2016','ShipRSImageNet'])
parser.add_argument('--bin_granularity', type=int, default=10)
args = parser.parse_args() 

def main(args):
	annot_files = sorted(np.array(os.listdir(args.annotation_path)))
	hist = histogram_calc(args, annot_files)
	#print(hist.sum(axis=0))
	
	plt.figure(figsize=(7,6))
	plt.vlines(np.linspace(0,170,18), 0, hist.sum(axis=0), linewidth=20)
	plt.xticks(np.linspace(0,170,18), fontsize=10)
	plt.yticks()
	plt.xlabel('Orientations (deg)', fontsize=15)
	plt.ylabel('# Objects', fontsize=15)
	plt.savefig('angle_dist_DOTA_ISO')
	plt.close()
	

if __name__ == '__main__':
	main(args)

"""
plt.figure(figsize=(5,5))
plt.hist(angles_corr, bins=18)
plt.xlabel('Orientation (deg)')
plt.ylabel('# Objects')
plt.savefig('angle_histogram_DOTA')
plt.close()


for i in np.unique(classes):
	plt.figure(figsize=(5,5))
	plt.hist(angles[np.where(classes==i)], bins=50)
	plt.xlabel('Orientation (deg)')
	plt.ylabel('# Objects')
	plt.title(f'{class_names[int(i)]} class angle histogram')
	plt.savefig(f'angle_histograms_per_class_best/{class_names[int(i)]}_hist_1.jpg')
	plt.close()
"""