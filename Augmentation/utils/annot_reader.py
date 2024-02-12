import xml.etree.ElementTree as ET
import numpy as np
import os

HRSC_classes_dict = {1:'100000001' , 2:'100000002' , 3:'100000003' , 4:'100000004' , 5:'100000005' ,
					 6:'100000006' , 7:'100000007' , 8:'100000008' , 9:'100000009' , 10:'100000010',
					 11:'100000011', 12:'100000012', 13:'100000013', 14:'100000014', 15:'100000015',
					 16:'100000016', 17:'100000017', 18:'100000018', 19:'100000019', 20:'100000020',
					 21:'100000022', 22:'100000024', 23:'100000025', 24:'100000026', 25:'100000027',
					 26:'100000028', 27:'100000029', 28:'100000030', 29:'100000031', 30:'100000032',
					 31:'100000033'}

DOTA_v1_5_classes_dict = {1:'plane', 2:'ship', 3:'storage-tank', 4:'baseball-diamond', 5:'tennis-court', 6:'basketball-court', 
						  7:'ground-track-field', 8:'harbor', 9:'bridge', 10:'large-vehicle', 11:'small-vehicle', 12:'helicopter',
						  13:'roundabout', 14:'soccer-ball-field', 15:'swimming-pool', 16:'container-crane'}

HRSC_classes_dict_inv = {v: k for k, v in HRSC_classes_dict.items()}
DOTA_v1_5_classes_dict_inv = {v: k for k, v in DOTA_v1_5_classes_dict.items()}


def annot_obj_reader(path, dataset_type):
	if dataset_type=='DOTA_v1.5':
		fil = open(path, 'r')
		lines = fil.readlines()
		if lines==[]:
			return []
		else:
			for j, line in enumerate(lines):
				
				#coors_hbb = np.array(line.split(" ")[:8]).astype(float)
				items = line.split(" ")
				cl = items[8]
				dif = int(items[9])
				Class_id = DOTA_v1_5_classes_dict_inv[cl]
				x1 = float(items[0])
				y1 = float(items[1])
				x2 = float(items[2])
				y2 = float(items[3])
				x3 = float(items[4])
				y3 = float(items[5])
				x4 = float(items[6])
				y4 = float(items[7])
				if (j==0):
					obj_info = np.array([Class_id,x1,y1,x2,y2,x3,y3,x4,y4]).reshape(1,9)
				else:
					obj_info = np.vstack([obj_info, [Class_id,x1,y1,x2,y2,x3,y3,x4,y4]])
			return obj_info

	elif dataset_type=='HRSC2016':
		tree = ET.parse(path)
		root = tree.getroot()
		objs = root.find('HRSC_Objects').findall('HRSC_Object')
		if objs==[]:
			return []
		else:
			for i, obj in enumerate(objs):
				cl = HRSC_classes_dict_inv[obj.find('Class_ID').text]
				if (cl==31):
					print('THERE IS OBJ')
				cx = float(obj.find('mbox_cx').text)
				cy = float(obj.find('mbox_cy').text)
				w = float(obj.find('mbox_w').text)
				h = float(obj.find('mbox_h').text)
				ang = float(obj.find('mbox_ang').text)
				if i==0:
					obj_info = np.array([cl, cx, cy, w, h, ang]).reshape(1,6)
				else:
					obj_info = np.vstack([obj_info, [cl, cx, cy, w, h, ang]])
			return obj_info
	
	elif dataset_type=='ShipRSImageNet':
		tree = ET.parse(path)
		root = tree.getroot()
		objs = root.findall('object')
		if objs==[]:
			return []
		else:
			for i, obj in enumerate(objs):
				cl = int(obj.find('level_3').text)
				x1 = float(obj.find('polygon').find('x1').text)
				y1 = float(obj.find('polygon').find('y1').text)
				x2 = float(obj.find('polygon').find('x2').text)
				y2 = float(obj.find('polygon').find('y2').text)
				x3 = float(obj.find('polygon').find('x3').text)
				y3 = float(obj.find('polygon').find('y3').text)
				x4 = float(obj.find('polygon').find('x4').text)
				y4 = float(obj.find('polygon').find('y4').text)
				if i==0:
					obj_info = np.array([cl,x1,y1,x2,y2,x3,y3,x4,y4]).reshape(1,9)
				else:
					obj_info = np.vstack([obj_info, [cl,x1,y1,x2,y2,x3,y3,x4,y4]])
			return obj_info

	else:
		print(f'{dataset_type} is not a valid dataset option')
		return False

def annot_reader_and_remover(dataset_type, test_dir, new_dir, ann_name, angle, quant):
	if dataset_type=='HRSC2016':
		test_path = os.path.join(test_dir, ann_name)
		new_path = os.path.join(new_dir, ann_name)
		tree = ET.parse(test_path)
		root = tree.getroot()
		objs = root.find('HRSC_Objects')
		
		nodes_to_remove=[]
		for obj in objs.findall('HRSC_Object'):
			ang = float(obj.find('mbox_ang').text)
			if ((ang*180/np.pi+90)%quant)>(quant/2):
				round_angle = int((ang*180/np.pi+90)//quant*quant+quant)
			else:
				round_angle = int((ang*180/np.pi+90)//quant*quant)
			if round_angle==180:
				round_angle=0
			
			if round_angle!=angle:
				nodes_to_remove.append(obj)

		for obj in nodes_to_remove:
			objs.remove(obj)

		tree.write(new_path)
		