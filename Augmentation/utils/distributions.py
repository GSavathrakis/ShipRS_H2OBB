import os
import numpy as np 
from .xml_reader import annot_obj_reader

def histogram_calc(args, annotation_filenames):
	if args.dataset_type=='HRSC2016':
		orientation_histogram = np.zeros((31, 180//args.bin_granularity))
	elif args.dataset_type=='ShipRSImageNet':
		orientation_histogram = np.zeros((50, 180//args.bin_granularity))
	
	for i in range(0,len(annotation_filenames)):
		fil_path = os.path.join(args.annotation_path, annotation_filenames[i])
		objs_info = annot_obj_reader(fil_path, args.dataset_type)
		if objs_info==[]:
			continue
		if args.dataset_type=='HRSC2016':
			round_angles=[]
			for k in range(len(objs_info)):
				if ((objs_info[k,5]*180/np.pi+90)%args.bin_granularity)>(args.bin_granularity/2):
					round_angle = (objs_info[k,5]*180/np.pi+90)//args.bin_granularity*args.bin_granularity+args.bin_granularity
				else:
					round_angle = (objs_info[k,5]*180/np.pi+90)//args.bin_granularity*args.bin_granularity
				round_angles.append(round_angle)
			round_angles = np.array(round_angles)
			round_angles[np.where(round_angles==180)]=0
		
		elif args.dataset_type=='ShipRSImageNet':
			round_angles=[]
			for k in range(len(objs_info)):
				xs = objs_info[k,1::2]
				ys = objs_info[k,2::2]

				side01 = np.sqrt((xs[1]-xs[0])**2+(ys[1]-ys[0])**2)
				side12 = np.sqrt((xs[2]-xs[1])**2+(ys[2]-ys[1])**2)

				ymin = ys.min()
				xmax = xs.max()
				x_ymin = xs[np.where(ys==ymin)]
				y_xmax = ys[np.where(xs==xmax)]
				if len(y_xmax)>1:
					y_xmax = y_xmax[np.where(y_xmax!=ymin)]
					if len(y_xmax)>1:
						y_xmax = y_xmax[np.where(np.abs(y_xmax-ymin)==np.abs(y_xmax-ymin).min())]
				if len(x_ymin)>1:
					x_ymin = x_ymin[np.where(x_ymin!=xmax)]
					if len(x_ymin)>1:
						x_ymin = x_ymin[np.where(np.abs(x_ymin-xmax)==np.abs(x_ymin-xmax).min())]
				side1 = np.sqrt((y_xmax-ymin)**2+(xmax-x_ymin)**2)
				if side1==side01:
					side2 = side12
				else:
					side2 = side01

				if side1>=side2:
					angle = np.pi/2 + np.arctan2(y_xmax-ymin, xmax-x_ymin)
				else:
					angle = np.arctan2(y_xmax-ymin, xmax-x_ymin)
				
				if ((angle*180/np.pi)%args.bin_granularity)>(args.bin_granularity/2):
					round_angle = (angle*180/np.pi)//args.bin_granularity*args.bin_granularity+args.bin_granularity
				else:
					round_angle = (angle*180/np.pi)//args.bin_granularity*args.bin_granularity
				
				round_angles.append(round_angle[0])
			round_angles=np.array(round_angles)
			round_angles[np.where(round_angles==180)]=0
		
		for j in range(len(objs_info)):
			orientation_histogram[int(objs_info[j][0]-1)][int(round_angles[j]//args.bin_granularity)]+=1

	return orientation_histogram