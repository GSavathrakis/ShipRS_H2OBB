import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import cv2
import os
import copy
import xml.etree.ElementTree as ET
import random
import re

from .annot_reader import annot_obj_reader
from .distributions import histogram_calc

HRSC_classes_dict = {1:'100000001' , 2:'100000002' , 3:'100000003' , 4:'100000004' , 5:'100000005' ,
					 6:'100000006' , 7:'100000007' , 8:'100000008' , 9:'100000009' , 10:'100000010',
					 11:'100000011', 12:'100000012', 13:'100000013', 14:'100000014', 15:'100000015',
					 16:'100000016', 17:'100000017', 18:'100000018', 19:'100000019', 20:'100000020',
					 21:'100000022', 22:'100000024', 23:'100000025', 24:'100000026', 25:'100000027',
					 26:'100000028', 27:'100000029', 28:'100000030', 29:'100000031', 30:'100000032',
					 31:'100000033'}

ShipRSImageNet_classes_dict = {1:'Other Ship', 2:'Other Warship', 3:'Submarine', 4:'Other Aircraft Carrier',
					5:'Enterprise', 6:'Nimitz', 7:'Midway', 8:'Ticonderoga',
					9:'Other Destroyer', 10:'Atago DD', 11:'Arleigh Burke DD', 12:'Hatsuyuki DD',
					13:'Hyuga DD', 14:'Asagiri DD', 15:'Other Frigate', 16:'Perry FF', 17:'Patrol',
					18:'Other Landing', 19:'YuTing LL', 20:'YuDeng LL',
					21:'YuDao LL', 22:'YuZhao LL', 23:'Austin LL', 24:'Osumi LL',
					25:'Wasp LL', 26:'LSD 41 LL', 27:'LHA LL', 28:'Commander',
					29:'Other Auxiliary Ship', 30:'Medical Ship', 31:'Test Ship',
					32:'Training Ship', 33:'AOE', 34:'Masyuu AS', 35:'Sanantonio AS',
					36:'EPF', 37:'Other Merchant', 38:'Container Ship', 39:'RoRo',
					40:'Cargo', 41:'Barge', 42:'Tugboat', 43:'Ferry', 44:'Yacht',
					45:'Sailboat', 46:'Fishing Vessel', 47:'Oil Tanker', 48:'Hovercraft',
					49:'Motorboat', 50:'Dock'}

DOTA_v1_5_classes_dict = {1:'plane', 2:'ship', 3:'storage-tank', 4:'baseball-diamond', 5:'tennis-court', 6:'basketball-court', 
						  7:'ground-track-field', 8:'harbor', 9:'bridge', 10:'large-vehicle', 11:'small-vehicle', 12:'helicopter',
						  13:'roundabout', 14:'soccer-ball-field', 15:'swimming-pool', 16:'container-crane'}

Classes_Dict = [DOTA_v1_5_classes_dict, HRSC_classes_dict, ShipRSImageNet_classes_dict]	

class Augmenter:
	def __init__(self, args):
		self.args = args

	def augment(self, histogram, annot_files, class_files):
		cp_annot_files = copy.deepcopy(annot_files)
		if self.args.augm_method=='SSO':
			n_objs = np.sum(histogram)
			upper_obj_bound = round(n_objs*self.args.bin_granularity/180)
			curr_histogram = np.zeros(shape=(180//self.args.bin_granularity))

			objs_added=0
			while objs_added<n_objs:
				fils_max_n = cp_annot_files[np.where(cp_annot_files[:,1].astype(int)==cp_annot_files[:,1].astype(int).max())][:,0]
				fil_selected = random.choice(fils_max_n)
				fil_hist = histogram_calc(self.args, [fil_selected]).sum(axis=0)
				if objs_added==0:
					curr_histogram+=fil_hist
					if self.args.dataset_type=='DOTA_v1.5':
						img_sh, img_name, old_img_sh = self.rotate_img(os.path.join(self.args.image_path, fil_selected.split('/')[-1][:-4]+'.png'), 0)
						self.rotate_annots(fil_selected, 0, img_sh, old_img_sh)
					else:
						img_sh, img_name, _ = self.rotate_img(os.path.join(self.args.image_path, fil_selected.split('/')[-1][:-4]+'.bmp'), 0)
						self.rotate_annots(fil_selected, 0, img_sh)
					

				else:
					rot_ang, Vars = self.uniformity_check(curr_histogram, fil_hist)
					curr_histogram += np.roll(fil_hist, rot_ang//self.args.bin_granularity)
					if self.args.dataset_type=='DOTA_v1.5':
						img_sh, img_name, old_img_sh = self.rotate_img(os.path.join(self.args.image_path, fil_selected.split('/')[-1][:-4]+'.png'), -rot_ang)
						self.rotate_annots(fil_selected, -rot_ang, img_sh, old_img_sh)
					else:
						img_sh, img_name, _ = self.rotate_img(os.path.join(self.args.image_path, fil_selected.split('/')[-1][:-4]+'.bmp'), -rot_ang)
						self.rotate_annots(fil_selected, -rot_ang, img_sh)
					
				objs_added=int(curr_histogram.sum())
				cp_annot_files = np.delete(cp_annot_files, np.where(cp_annot_files[:,0]==fil_selected), axis=0)
			
		elif self.args.augm_method=='ISO':
			angs_init = np.zeros(len(cp_annot_files))
			files_init = []
			for i in range(len(cp_annot_files)):
				if self.args.dataset_type=='DOTA_v1.5':
					ann_name_num = cp_annot_files[i][0].split('/')[-1][:-4]+'_1.txt'
					img_name_num_old = cp_annot_files[i][0].split('/')[-1][:-4]+'.png'
					img_name_num_new = cp_annot_files[i][0].split('/')[-1][:-4]+'_1.png'
				else:
					ann_name_num = cp_annot_files[i][0].split('/')[-1][:-4]+'_1.xml'
					img_name_num_old = cp_annot_files[i][0].split('/')[-1][:-4]+'.bmp'
					img_name_num_new = cp_annot_files[i][0].split('/')[-1][:-4]+'_1.bmp'
				files_init = np.append(files_init, img_name_num_new[:-6])
				os.system(f'cp {cp_annot_files[i][0]} {os.path.join(self.args.aug_annotation_path, ann_name_num)}')
				os.system(f'cp {os.path.join(self.args.image_path, img_name_num_old)} {os.path.join(self.args.aug_image_path, img_name_num_new)}')

			files_init = np.array(files_init)

			upper_obj_bound = np.sum(histogram, axis=0).max()
			ind_max = np.where(np.sum(histogram, axis=0)==upper_obj_bound)[0][0]
			angle_max = ind_max*self.args.bin_granularity
			curr_histogram = copy.deepcopy(np.sum(histogram, axis=0))

			objs_added = np.sum(histogram)
			fils_used = []
			print('Augmenting phase 1')
			while curr_histogram[ind_max]<2*upper_obj_bound:
				fil_selected = random.choice(cp_annot_files[:,0])
				fil_hist = histogram_calc(self.args, [fil_selected]).sum(axis=0)
				rand_obj_ind = np.random.choice(np.where(fil_hist!=0)[0])
				rot_ang = (ind_max - rand_obj_ind)*self.args.bin_granularity
				if rot_ang not in angs_init[np.where(files_init==fil_selected.split('/')[-1][:-4])]:
					curr_histogram += np.roll(fil_hist, rot_ang//self.args.bin_granularity)
					if self.args.dataset_type=='DOTA_v1.5':
						img_sh, img_name, old_img_sh = self.rotate_img(os.path.join(self.args.image_path, fil_selected.split('/')[-1][:-4]+'.png'), -rot_ang)
						self.rotate_annots(fil_selected, -rot_ang, img_sh, old_img_sh)
					else:
						img_sh, img_name, _ = self.rotate_img(os.path.join(self.args.image_path, fil_selected.split('/')[-1][:-4]+'.bmp'), -rot_ang)
						self.rotate_annots(fil_selected, -rot_ang, img_sh)
					
					files_init = np.append(files_init, fil_selected.split('/')[-1][:-4])
					angs_init = np.append(angs_init, rot_ang)
					objs_added=int(curr_histogram.sum())

			new_upper_obj_bound = curr_histogram.max()
			print('Augmenting phase 2')
			while objs_added<2*upper_obj_bound*(180//self.args.bin_granularity):
				print(objs_added)
				fil_selected = random.choice(cp_annot_files[:,0])
				fil_hist = histogram_calc(self.args, [fil_selected]).sum(axis=0)
				rot_ang, Vars = self.uniformity_check(curr_histogram, fil_hist)
				if ((curr_histogram + np.roll(fil_hist, rot_ang//self.args.bin_granularity)).max()<=new_upper_obj_bound and rot_ang not in angs_init[np.where(files_init==fil_selected.split('/')[-1][:-4])]):
					curr_histogram += np.roll(fil_hist, rot_ang//self.args.bin_granularity)
					if self.args.dataset_type=='DOTA_v1.5':
						img_sh, img_name, old_img_sh = self.rotate_img(os.path.join(self.args.image_path, fil_selected.split('/')[-1][:-4]+'.png'), -rot_ang)
						self.rotate_annots(fil_selected, -rot_ang, img_sh, old_img_sh)
					else:
						img_sh, img_name, _ = self.rotate_img(os.path.join(self.args.image_path, fil_selected.split('/')[-1][:-4]+'.bmp'), -rot_ang)
						self.rotate_annots(fil_selected, -rot_ang, img_sh)
					
					files_init = np.append(files_init, fil_selected.split('/')[-1][:-4])
					angs_init = np.append(angs_init, rot_ang)
					objs_added=int(curr_histogram.sum())
				else:
					continue
			
		
		elif self.args.augm_method=='ISC':
			angs_init = np.zeros(len(cp_annot_files))
			files_init = []
			for i in range(len(cp_annot_files)):
				ann_name_num = cp_annot_files[i][0].split('/')[-1][:-4]+'_1.xml'
				img_name_num_old = cp_annot_files[i][0].split('/')[-1][:-4]+'.bmp'
				img_name_num_new = cp_annot_files[i][0].split('/')[-1][:-4]+'_1.bmp'
				files_init = np.append(files_init, img_name_num_new[:-6])
				os.system(f'cp {cp_annot_files[i][0]} {os.path.join(self.args.aug_annotation_path, ann_name_num)}')
				os.system(f'cp {os.path.join(self.args.image_path, img_name_num_old)} {os.path.join(self.args.aug_image_path, img_name_num_new)}')

			files_init = np.array(files_init)

			if self.args.dataset_type=='HRSC2016':
				ind_dict = 0
			elif self.args.dataset_type=='ShipRSImageNet':
				ind_dict = 1
			
			for cl in range(0,histogram.shape[0]):
				print(f'Augmenting objects of class {Classes_Dict[ind_dict][cl+1]}')
				fils_cl=[]
				for im in range(len(class_files)):
					if (len(np.unique(class_files[im][1]))==1) and (np.unique(class_files[im][1][0])==cl+1):
						fils_cl.append(class_files[im][0])
				if fils_cl==[]:
					print(f'No files with objects belonging only to class {Classes_Dict[ind_dict][cl+1]}')
				else:
					print(f'{len(fils_cl)} files with objects belonging only to class {Classes_Dict[ind_dict][cl+1]}')
					
					cl_hist = histogram[cl]
					upper_cl_obj_bound = cl_hist.max()
					curr_histogram = copy.deepcopy(cl_hist)
					objs_added = np.sum(cl_hist)
					cnt=0
					while objs_added<upper_cl_obj_bound*(180//self.args.bin_granularity):
						fil_selected = random.choice(fils_cl)
						fil_hist = histogram_calc(self.args, [fil_selected]).sum(axis=0)
						rot_ang, Vars = self.uniformity_check(curr_histogram, fil_hist)
						
						if ((curr_histogram + np.roll(fil_hist, rot_ang//self.args.bin_granularity)).max()<=upper_cl_obj_bound and rot_ang not in angs_init[np.where(files_init==fil_selected.split('/')[-1][:-4])]):
							curr_histogram += np.roll(fil_hist, rot_ang//self.args.bin_granularity)
							img_sh, img_name = self.rotate_img(os.path.join(self.args.image_path, fil_selected.split('/')[-1][:-4]+'.bmp'), -rot_ang)
							self.rotate_annots(fil_selected, -rot_ang, img_sh)
							files_init = np.append(files_init, fil_selected.split('/')[-1][:-4])
							angs_init = np.append(angs_init, rot_ang)
							objs_added=int(curr_histogram.sum())
						else:
							cnt+=1
							if (cnt==len(fils_cl)*(180//self.args.bin_granularity)):
								break

	def uniformity_check(self, old_hist, new_hist):
		Vars=[]
		for i in range(0, 180//self.args.bin_granularity):
			cand_hist = old_hist + np.roll(new_hist, i)
			if i==0:
				min_var = np.var(cand_hist)
				rot_ang = i*self.args.bin_granularity
			else:
				if np.var(cand_hist)<min_var:
					min_var = np.var(cand_hist)
					rot_ang = i*self.args.bin_granularity
			Vars.append(np.var(cand_hist))
		Vars = np.array(Vars)
		return rot_ang, Vars

	def rotate_img(self, img_filename, ang):
		img = cv2.imread(img_filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_rot = rotate(img, ang)
		if self.args.augm_method=='SSO':
			img_name = img_filename.split('/')[-1]
		elif ((self.args.augm_method=='ISO') or (self.args.augm_method=='ISC')):
			old_image_name = img_filename.split('/')[-1]
			imgs_in_aug_dir = np.array(sorted(os.listdir(self.args.aug_image_path), key=self.custom_sort_key))
			for img_aug in imgs_in_aug_dir:
				if old_image_name[:-4] == img_aug[:-6]:
					img_name = img_aug[:-5] + str(int(img_aug[-5])+1) + img_aug[-4:]
				elif old_image_name[:-4] == img_aug[:-7]:
					img_name = img_aug[:-6] + str(int(img_aug[-6:-4])+1) + img_aug[-4:]
		cv2.imwrite(os.path.join(self.args.aug_image_path, img_name) , cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB))
		
		return img_rot.shape, img_name, img.shape

	def rotate_annots(self, annot_filename, ang, img_shape, prev_img_shape=None):
		
		if self.args.augm_method=='SSO':
			annot_name = annot_filename.split('/')[-1]
		elif ((self.args.augm_method=='ISO') or (self.args.augm_method=='ISC')):
			old_annot_name = annot_filename.split('/')[-1]
			annots_in_dir = np.array(sorted(os.listdir(self.args.aug_annotation_path), key=self.custom_sort_key))
			for annot_aug in annots_in_dir:
				if old_annot_name[:-4] == annot_aug[:-6]:
					annot_name = annot_aug[:-5] + str(int(annot_aug[-5])+1) + annot_aug[-4:]
				elif old_annot_name[:-4] == annot_aug[:-7]:
					annot_name = annot_aug[:-6] + str(int(annot_aug[-6:-4])+1) + annot_aug[-4:]


		path_aug = os.path.join(self.args.aug_annotation_path, annot_name)
		os.system(f'cp {annot_filename} {path_aug}')
		ang_rads = ang*np.pi/180
		
		if self.args.dataset_type == 'HRSC2016':
			tree = ET.parse(path_aug)
			root = tree.getroot()
			W = float(root.find('Img_SizeWidth').text)
			H = float(root.find('Img_SizeHeight').text)
			objs = root.find('HRSC_Objects').findall('HRSC_Object')
			for obj in objs:
				cx = float(obj.find('mbox_cx').text)
				cy = float(obj.find('mbox_cy').text)
				prev_ang = float(obj.find('mbox_ang').text)

				cx_new = np.cos(ang_rads)*(cx-W/2) - np.sin(ang_rads)*(H/2-cy) + img_shape[1]/2
				cy_new = -(np.sin(ang_rads)*(cx-W/2) + np.cos(ang_rads)*(H/2-cy) - img_shape[0]/2)

				ang_new = prev_ang - ang_rads
				if ang_new < -np.pi/2:
					ang_new += np.pi
				elif ang_new > np.pi/2:
					ang_new -= np.pi

				obj.find('mbox_cx').text = str(cx_new)
				obj.find('mbox_cy').text = str(cy_new)
				obj.find('mbox_ang').text = str(ang_new)
			root.find('Img_SizeWidth').text = str(img_shape[1])
			root.find('Img_SizeHeight').text = str(img_shape[0])
			tree.write(path_aug)
		
		elif self.args.dataset_type=='ShipRSImageNet':
			tree = ET.parse(path_aug)
			root = tree.getroot()
			W = float(root.find('size').find('width').text)
			H = float(root.find('size').find('height').text)
			objs = root.findall('object')
			for obj in objs:
				x1 = float(obj.find('polygon').find('x1').text)
				y1 = float(obj.find('polygon').find('y1').text)
				x2 = float(obj.find('polygon').find('x2').text)
				y2 = float(obj.find('polygon').find('y2').text)
				x3 = float(obj.find('polygon').find('x3').text)
				y3 = float(obj.find('polygon').find('y3').text)
				x4 = float(obj.find('polygon').find('x4').text)
				y4 = float(obj.find('polygon').find('y4').text)

				x1_new = np.cos(ang_rads)*(x1-W/2) - np.sin(ang_rads)*(H/2-y1) + img_shape[1]/2
				y1_new = -(np.sin(ang_rads)*(x1-W/2) + np.cos(ang_rads)*(H/2-y1) - img_shape[0]/2)
				x2_new = np.cos(ang_rads)*(x2-W/2) - np.sin(ang_rads)*(H/2-y2) + img_shape[1]/2
				y2_new = -(np.sin(ang_rads)*(x2-W/2) + np.cos(ang_rads)*(H/2-y2) - img_shape[0]/2)
				x3_new = np.cos(ang_rads)*(x3-W/2) - np.sin(ang_rads)*(H/2-y3) + img_shape[1]/2
				y3_new = -(np.sin(ang_rads)*(x3-W/2) + np.cos(ang_rads)*(H/2-y3) - img_shape[0]/2)
				x4_new = np.cos(ang_rads)*(x4-W/2) - np.sin(ang_rads)*(H/2-y4) + img_shape[1]/2
				y4_new = -(np.sin(ang_rads)*(x4-W/2) + np.cos(ang_rads)*(H/2-y4) - img_shape[0]/2)

				obj.find('polygon').find('x1').text = str(x1_new)
				obj.find('polygon').find('y1').text = str(y1_new)
				obj.find('polygon').find('x2').text = str(x2_new)
				obj.find('polygon').find('y2').text = str(y2_new)
				obj.find('polygon').find('x3').text = str(x3_new)
				obj.find('polygon').find('y3').text = str(y3_new)
				obj.find('polygon').find('x4').text = str(x4_new)
				obj.find('polygon').find('y4').text = str(y4_new)
			root.find('size').find('width').text = str(img_shape[1])
			root.find('size').find('height').text = str(img_shape[0])
			tree.write(path_aug)

		elif self.args.dataset_type=='DOTA_v1.5':
			assert prev_img_shape!=None

			H = prev_img_shape[0]
			W = prev_img_shape[1]

			with open(path_aug, 'r') as fil:
				lines = fil.readlines()

			new_lines = []
			for i, line in enumerate(lines):
				if i<2:
					new_lines.append(line)
				else:
					items = line.split(" ")
					x1 = float(items[0])
					y1 = float(items[1])
					x2 = float(items[2])
					y2 = float(items[3])
					x3 = float(items[4])
					y3 = float(items[5])
					x4 = float(items[6])
					y4 = float(items[7])

					x1_new = np.cos(ang_rads)*(x1-W/2) - np.sin(ang_rads)*(H/2-y1) + img_shape[1]/2
					y1_new = -(np.sin(ang_rads)*(x1-W/2) + np.cos(ang_rads)*(H/2-y1) - img_shape[0]/2)
					x2_new = np.cos(ang_rads)*(x2-W/2) - np.sin(ang_rads)*(H/2-y2) + img_shape[1]/2
					y2_new = -(np.sin(ang_rads)*(x2-W/2) + np.cos(ang_rads)*(H/2-y2) - img_shape[0]/2)
					x3_new = np.cos(ang_rads)*(x3-W/2) - np.sin(ang_rads)*(H/2-y3) + img_shape[1]/2
					y3_new = -(np.sin(ang_rads)*(x3-W/2) + np.cos(ang_rads)*(H/2-y3) - img_shape[0]/2)
					x4_new = np.cos(ang_rads)*(x4-W/2) - np.sin(ang_rads)*(H/2-y4) + img_shape[1]/2
					y4_new = -(np.sin(ang_rads)*(x4-W/2) + np.cos(ang_rads)*(H/2-y4) - img_shape[0]/2)


					new_line = str(f'{str(x1_new)} {str(y1_new)} {str(x2_new)} {str(y2_new)} {str(x3_new)} {str(y3_new)} {str(x4_new)} {str(y4_new)} {items[8]} {items[9]}')
					new_lines.append(new_line)

			with open(path_aug, 'w') as fil_n:
				fil_n.writelines(new_lines)




		

	def custom_sort_key(self, filename):
		match = re.search(r'_(\d+)\.', filename)
		if match:
			return int(match.group(1))
		return filename

