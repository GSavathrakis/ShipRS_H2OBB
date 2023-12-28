import numpy as np 
import matplotlib.pyplot as plt 

Classes = np.load('../OBB_generation/gt_classes.npy', allow_pickle=True)
Classes_in_IOU = np.loadtxt('../OBB_generation/Classes_n_Angles.csv', delimiter=',')[:,0]

Classes_cor = []
for im in Classes:
	for i in range(0,len(im)):
		Classes_cor.append(im[i])

Classes_cor = np.array(Classes_cor).astype(int)
unique, counts = np.unique(Classes_cor, return_counts=True)
num_classes_gt = np.asarray((unique, counts)).T.astype(float)
num_classes_in_IOU = np.asarray((np.unique(Classes_in_IOU, return_counts=True))).T.astype(float)

num_classes_gt[:,1] = num_classes_gt[:,1]/np.sum(num_classes_gt[:,1])*100
num_classes_in_IOU[:,1] = num_classes_in_IOU[:,1]/np.sum(num_classes_in_IOU[:,1])*100


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(8,4))

ax1.bar(num_classes_gt[:,0], num_classes_gt[:,1], color='blue')
ax1.set_xlabel('class')
ax1.set_ylabel('class percentage')
ax1.set_title('GT class distribution')

ax2.bar(num_classes_in_IOU[:,0], num_classes_in_IOU[:,1], color='maroon')
ax2.set_xlabel('class')
ax2.set_ylabel('class percentage')
ax2.set_title('Objects in IOU class distribution')

plt.show()