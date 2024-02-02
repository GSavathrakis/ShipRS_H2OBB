import torch
from tqdm import tqdm
import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt 
from matplotlib.path import Path
import matplotlib.patches as patches
import argparse
import copy

from segment_anything import sam_model_registry, SamPredictor

from utils.annot_utils import file_OBB_reader, create_files
from utils import bbox_utils


parser = argparse.ArgumentParser('Set segment anything model', add_help=False)
parser.add_argument('--dataset', type=str, choices=['HRSC2016', 'ShipRSImageNet', 'DOTA'])
parser.add_argument('--image_path', type=str)
parser.add_argument('--annotation_path', type=str)
parser.add_argument('--sam_checkpoint_path', type=str)
parser.add_argument('--new_annotations_path', type=str)
parser.add_argument('--IOU_thres', type=float, default=0.7)
parser.add_argument('--kernel_size_perc', type=float, default=0.03)
parser.add_argument('--kernel_type', type=str, default='ellipsoid')
parser.add_argument('--n_points', type=int, default=5)
parser.add_argument('--image_vis', type=bool, default=False)
parser.add_argument('--images_with_masks_path', type=str)
parser.add_argument('--images_with_boxes_path', type=str)
parser.add_argument('--gen_mode', action='store_true')

args = parser.parse_args() 

def mask_predictor_one_obj(predictor, input_p, input_l, input_box):
	mask, score, _  = predictor.predict(
		point_coords=input_p,
		point_labels=input_l,
		box=input_box,
		multimask_output=False,
	)
	return mask, score


def run_plots_nopoints(image, img_name, masks, gt_BBs, mask_BBs, mask_RBBs, output_mask_dir, output_box_dir, OBBs=False):
	plt.figure()
	plt.imshow(image)
	
	for mask in masks:
		show_mask(mask, plt.gca(), random_color=True)
	plt.axis('off')
	plt.savefig(f'{output_mask_dir}/{img_name[:-4]}.jpg')
	plt.close()
	"""
	for gt_bb in gt_BBs:
		show_box(gt_bb, plt.gca(), 'green')
	"""
	plt.figure()
	plt.imshow(image)
	if OBBs:
		for mask_rbb in mask_RBBs:
			show_rbox(mask_rbb, plt.gca(), 'red')
	else:
		for mask_bb in mask_BBs:
			show_box(mask_bb, plt.gca(), 'white')
	plt.axis('off')
	plt.savefig(f'{output_box_dir}/{img_name[:-4]}.jpg')
	plt.close()

	

def run_plots(image, img_name, points, masks, output_dir, with_masks=False):
	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	for point in points:
		show_points(point, np.array([1]), plt.gca())
	for mask in masks:
		show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
	plt.axis('off')
	plt.savefig(f'{output_dir}/{img_name[:-4]}.jpg')
	plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=125):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, edgecolor):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))   


def show_rbox(box, ax, edgecolor):
	verts=[]
	for pt in box:
		tup_pt = (pt[0], pt[1])
		verts.append(tup_pt)
	verts.append((box[0][0], box[0][1]))

	codes = [Path.MOVETO,
		 Path.LINETO,
		 Path.LINETO,
		 Path.LINETO,
		 Path.CLOSEPOLY,]

	path = Path(verts, codes)
	patch = patches.PathPatch(path, facecolor=None, edgecolor=edgecolor, fill=False, lw=2)
	ax.add_patch(patch)

def one_img_sam(args, image, labels, predictor):
	label_points = bbox_utils.box_cxcywh_to_cxcy(labels)
	labels_diags = bbox_utils.box_cxcywh_to_diag(labels, args.n_points)
	label_boxes = bbox_utils.box_cxcywh_to_xyxy(labels)
	predictor.set_image(image)


	masks=[]
	foreg1 = np.hstack((np.array([1]), np.ones((args.n_points//2)), np.zeros((args.n_points//2))))
	foreg2 = np.hstack((np.array([1]), np.zeros((args.n_points//2)), np.ones((args.n_points//2))))
	for i in range(len(labels_diags)):
		mask_obj_i_diag1, score_obj_i_diag1 = mask_predictor_one_obj(predictor, labels_diags[i], foreg1, label_boxes[i])
		mask_obj_i_diag2, score_obj_i_diag2 = mask_predictor_one_obj(predictor, labels_diags[i], foreg2, label_boxes[i])
		if (score_obj_i_diag2[0]>=score_obj_i_diag1[0]):
			masks.append(mask_obj_i_diag2)
			diag_direct = 'bltr'
		else:
			masks.append(mask_obj_i_diag1)
			diag_direct = 'tlbr'

	return masks, diag_direct

def angle_calc_for_IOU_thres(args, image, image_name , masks, diag_dir, labels, IOUs, angle_info, length_info, opening_ang_info, im_w, im_h, img_vis, thres, kern_perc, kern_type):
	mask_boxes = np.zeros((len(masks), 4))
	mask_rboxes=[]
	mask_hboxes=[]
	inds_no_masks=[]
	Cls=[]
	n=0
	for mask in masks:
		if ((np.where(mask[0]==True)[0].shape[0]==0.) or (np.where(mask[0]==True)[1].shape[0]==0.)):
			inds_no_masks.append(n)
		else:
			mask_boxes[n] = np.array([np.where(mask[0]==True)[1].min(), np.where(mask[0]==True)[0].min(), np.where(mask[0]==True)[1].max(), np.where(mask[0]==True)[0].max()])
		
		n+=1
	mask_boxes = mask_boxes.astype(int)
	ground_truth_boxes = bbox_utils.box_cxcywh_to_xyxy(labels).astype(int)
	for j in range(len(ground_truth_boxes)):
		if j not in inds_no_masks:
			l_gt = np.sqrt(labels[j][3]**2+labels[j][4]**2)
			box, hbox_j, angle, l, w = bbox_utils.rotated_BB_calculation(masks[j], diag_dir, ground_truth_boxes[j], kern_type, kern_perc*l_gt)
			mask_box_region = bbox_utils.box_masking(hbox_j, im_w, im_h)
			gt_box_region = bbox_utils.box_masking(ground_truth_boxes[j], im_w, im_h)
			IOU = bbox_utils.IOU_calc(mask_box_region, gt_box_region)
			if IOU>=thres:
				mask_rboxes.append(box)
				mask_hboxes.append(hbox_j)
				Cls.append(int(labels[j][0]))
				if (angle<0):
					angle = angle + 180
				angle_info.append(np.array([labels[j,0], angle]))
				length_info.append(l_gt)
				opening_ang_info.append(2*np.arctan2(0.5*w, 0.5*l)*180./np.pi)
			IOUs.append(IOU)
	
	if img_vis:
		run_plots_nopoints(image, image_name, masks, ground_truth_boxes, mask_boxes, mask_rboxes, args.images_with_masks_path, args.images_with_boxes_path, OBBs=True)
	

	return mask_rboxes, mask_hboxes, Cls


def multi_img_sam(args, image_filenames, annotation_filenames, predictor):
	if args.gen_mode:
		aug_dir_path_annot = args.new_annotations_path
		if not os.path.exists(aug_dir_path_annot):
			os.mkdir(aug_dir_path_annot)
	if args.image_vis:
		assert args.images_with_masks_path != None and args.images_with_boxes_path != None
		if not os.path.exists(args.images_with_masks_path):
			os.makedirs(args.images_with_masks_path)
		if not os.path.exists(args.images_with_boxes_path):
			os.makedirs(args.images_with_boxes_path)
		
	IOUs=[]
	angle_info = []
	length_info = []
	opening_ang_info = []
	classes=[]
	for i in tqdm(range(0,len(image_filenames)), desc='Segmented images'):
		image = cv2.imread(os.path.join(args.image_path, image_filenames[i]))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		labels, exist_obj, extr_dsc = file_OBB_reader(os.path.join(args.annotation_path, annotation_filenames[i]), args.dataset)
		if exist_obj==True:
			classes.append(labels[:,0])
			masks, diag_dir = one_img_sam(args, image, labels.astype(float), predictor)
			mask_rbb, mask_hbb, Cls = angle_calc_for_IOU_thres(args, image, image_filenames[i], masks, diag_dir, labels.astype(float), IOUs, angle_info, length_info, opening_ang_info, image.shape[1], image.shape[0], args.image_vis, args.IOU_thres, args.kernel_size_perc, args.kernel_type)
			if args.gen_mode:
				annot = os.path.join(aug_dir_path_annot, annotation_filenames[i])
				create_files(annot, image.shape[1], image.shape[0], mask_rbb, mask_hbb, extr_dsc, Cls, args.dataset)
		else:
			if args.gen_mode:
				os.system(f'cp {os.path.join(args.annotation_path, annotation_filenames[i])} {os.path.join(aug_dir_path_annot, annotation_filenames[i])}')
		
	angle_info = np.array(angle_info)
	return IOUs, angle_info, classes, length_info, opening_ang_info

def main(args):
	image_filenames = np.array(sorted(os.listdir(args.image_path))) # Image filenames in image folder
	annotation_filenames = np.array(sorted(os.listdir(args.annotation_path))) # Annotation filenames in annotations folder

	
	for i in range(len(image_filenames)):
		assert image_filenames[i][:-4] == annotation_filenames[i][:-4]

	#sam_checkpoint = "sam_vit_h_4b8939.pth"
	model_type = "vit_h"

	device = 'cuda' if torch.cuda.is_available() else 'cpu' #Select GPU if available, else use cpu

	sam = sam_model_registry[model_type](checkpoint = args.sam_checkpoint_path)
	sam.to(device=device)

	predictor = SamPredictor(sam)
	
	
	IOUs, Angles_per_class, Classes, lengths, openings = multi_img_sam(args, image_filenames, annotation_filenames, predictor)
	Classes = np.array(Classes).flatten()
	IOUs = np.array(IOUs)
	lengths = np.array(lengths)
	openings = np.array(openings)
	
	print(f'% objects with IOU>90%: {len(np.where(IOUs>=0.9)[0])/len(IOUs)*100}')
	print(f'% objects with IOU>80%: {len(np.where(IOUs>=0.8)[0])/len(IOUs)*100}')
	print(f'% objects with IOU>70%: {len(np.where(IOUs>=0.7)[0])/len(IOUs)*100}')
	print(f'% objects with IOU>60%: {len(np.where(IOUs>=0.6)[0])/len(IOUs)*100}')


	# Uncomment to save numpy arrays with object IoUs, class, and orientation distributions
	
	np.save('IOU', IOUs)
	np.save('gt_classes', Classes)
	np.savetxt('Classes_n_Angles_corr.csv', Angles_per_class, delimiter=',')
	
	
	
	
	
	


if __name__ == '__main__':
	main(args)