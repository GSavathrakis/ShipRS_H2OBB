import numpy as np 
import matplotlib.pyplot as plt 

cl_n_angs = np.loadtxt('../OBB_generation/Classes_n_Angles.csv', delimiter=',')
class_names = #np.loadtxt('Path to csv file with class names', dtype=str)
classes = cl_n_angs[:,0]
angles = cl_n_angs[:,1]


for i in np.unique(classes):
	plt.figure(figsize=(5,5))
	plt.hist(angles[np.where(classes==i)], bins=50)
	plt.xlabel('Orientation (deg)')
	plt.ylabel('# Objects')
	plt.title(f'{class_names[int(i)]} class angle histogram')
	plt.savefig(f'angle_histograms_per_class_best/{class_names[int(i)]}_hist_1.jpg')
	plt.close()