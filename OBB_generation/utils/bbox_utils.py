import numpy as np 
import cv2
import os
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt 
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def cv_contours_to_path(cv_contours):
    paths = []
    for contour in cv_contours:
        path_data = []
        for point in contour:
            x, y = point[0]
            path_data.append((Path.MOVETO if len(path_data) == 0 else Path.LINETO, (x, y)))
        path_data.append((Path.CLOSEPOLY, (0, 0)))
        codes, vertices = zip(*path_data)
        path = Path(vertices, codes)
        paths.append(path)
    return paths

def box_cxcywh_to_cxcy(labels):
	labels_cent = np.zeros(labels.shape)
	labels_cent[:,[1,3]] = labels[:,[1,3]]
	labels_cent[:,[2,4]] = labels[:,[2,4]]
	labels_cent = labels_cent.astype(int)
	labels_cent = labels_cent.reshape(labels_cent.shape[0],1,labels_cent.shape[1])
	return labels_cent[:,:,[1,2]]

def box_cxcywh_to_diag(labels, n_points):
	assert (n_points-1)%4==0
	pts_per_diag = n_points//4
	labels_diag = np.zeros((labels.shape[0],n_points,2))
	labels_diag[:,0,:] = labels[:,[1,2]] # Center
	for n in range(0, (n_points-1)//2):
		if n<n_points//4:
			labels_diag[:,n+1,:] = (labels[:,[1,2]] + (n+1)*(labels[:,[3,4]]/2)/(pts_per_diag+1))
		else:
			labels_diag[:,n+1,:] = (labels[:,[1,2]] - (n-pts_per_diag+1)*(labels[:,[3,4]]/2)/(pts_per_diag+1))
	for n in range((n_points-1)//2, n_points-1):
		if n<((n_points-1)//2)+n_points//4:
			labels_diag[:,n+1,:] = (labels[:,[1,2]] + (n - (n_points-1)//2 + 1)*(np.concatenate((labels[:,3].reshape(-1,1), -labels[:,4].reshape(-1,1)), axis=1)/2)/(pts_per_diag+1))
		else:
			labels_diag[:,n+1,:] = (labels[:,[1,2]] - (n - (n_points-1)//2 - pts_per_diag + 1)*(np.concatenate((labels[:,3].reshape(-1,1), -labels[:,4].reshape(-1,1)), axis=1)/2)/(pts_per_diag+1))

	labels_diag = labels_diag.astype(int)

	return labels_diag
	
def box_cxcywh_to_xyxy(labels):
	labels_res = np.zeros(labels.shape)
	labels_res[:,[1,3]] = labels[:,[1,3]]
	labels_res[:,[2,4]] = labels[:,[2,4]]
	labels_res = labels_res.astype(int)
	# In form x_min, y_min, x_max, y_max
	labels_xyxy = np.stack((labels_res[:,1]-labels_res[:,3]/2, labels_res[:,2]-labels_res[:,4]/2, labels_res[:,1]+labels_res[:,3]/2, labels_res[:,2]+labels_res[:,4]/2)).transpose()
	
	return labels_xyxy.astype(int)


def box_masking(coors, im_w, im_h):
	# Box oordinates are in format xmin, ymin, xmax, ymax
	bb_mask = np.zeros((im_h, im_w))
	bb_mask[coors[1]:coors[3], coors[0]:coors[2]]=1
	return bb_mask

def IOU_calc(box1, box2):
	assert box1.shape == box2.shape

	box1_TF = box1.astype(bool)
	box2_TF = box2.astype(bool)
	overlap = box1_TF*box2_TF
	union = box1_TF + box2_TF

	IOU = overlap.sum()/union.sum()

	return IOU

def rotated_BB_calculation(mask, diag_dir, gt_BB, kern_type, kern_size):
	mask_int = mask[0].astype(np.uint8)*255
	if kern_type=='square':
		kernel = np.ones((int(kern_size),int(kern_size)), np.uint8)
	elif kern_type=='circle':
		kernel = np.zeros((int(2**0.5*kern_size),int(2**0.5*kern_size)), np.uint8)
		kern_grid = np.indices(kernel.shape)
		sq_dist_grid = np.sqrt((kern_grid[0]-int(2**0.5*kern_size//2))**2+(kern_grid[1]-int(2**0.5*kern_size//2))**2)
		kernel[np.where(sq_dist_grid<=int(2**0.5*kern_size//2))]=1
	elif kern_type=='ellipsoid':
		if diag_dir=='tlbr':
			rot = np.arctan2(gt_BB[3]-gt_BB[1], gt_BB[2]-gt_BB[0]) + np.pi/2
		elif diag_dir=='bltr':
			rot = np.arctan2(gt_BB[3]-gt_BB[1], gt_BB[2]-gt_BB[0])
		
		a = 2**0.5*kern_size
		b = 2**0.5*kern_size//2

		w, h = int(2*a), int(2*a)
		center = (w//2, h//2)
		x = np.linspace(0, w-1, w)
		y = np.linspace(0, h-1, h)
 
		x, y = np.meshgrid(x, y)
		x_rot = (x - center[0]) * np.cos(rot) - (y - center[1]) * np.sin(rot) + center[0]
		y_rot = (x - center[0]) * np.sin(rot) + (y - center[1]) * np.cos(rot) + center[1]

		ellipse = (x_rot - center[0])**2 / (a**2) + (y_rot - center[1])**2 / (b**2)
		kernel = np.zeros((w, h))
		kernel[np.where(ellipse <= 1)] = 1
		kernel = kernel.astype(np.uint8) 



	mask_close = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel)
	contours, hierarchy = cv2.findContours(image=mask_close, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
	color = cv2.cvtColor(mask_close, cv2.COLOR_GRAY2RGB)
	cont_2d = max(contours, key=cv2.contourArea)[:,0,:]

	x_min = cont_2d[:,0].min()
	y_min = cont_2d[:,1].min()
	x_max = cont_2d[:,0].max()
	y_max = cont_2d[:,1].max()

	rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
	box = cv2.boxPoints(rect)
	box = np.int0(box)


	side32 = np.round(np.sqrt((box[3][0]-box[2][0])**2+(box[3][1]-box[2][1])**2)).astype(int)
	side03 = np.round(np.sqrt((box[3][0]-box[0][0])**2+(box[3][1]-box[0][1])**2)).astype(int)

	if (side32>=side03):
		angle = rect[2]
		length = side32
		width = side03
	else:
		angle = rect[2]+90
		length = side03
		width = side32
	
	return box, np.array([x_min, y_min, x_max, y_max]), angle, length, width