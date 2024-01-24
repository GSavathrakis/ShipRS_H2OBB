import xml.etree.ElementTree as ET
from xml.dom import minidom
from math import *
import numpy as np

DOTA_cls = np.array(['plane','ship','storage-tank','baseball-diamond','tennis-court','basketball-court','ground-track-field','harbor','bridge','large-vehicle','small-vehicle','helicopter','roundabout','soccer-ball-field','swimming-pool','container-crane'])

def annot_angles(annotation_file):
	annot_file = open(annotation_file, 'r')
	tree = ET.parse(annot_file)
	root = tree.getroot()
	Angles=[annotation_file]
	objs = root.find('HRSC_Objects').findall('HRSC_Object')
	for j in objs:
		ang = j.find('mbox_ang').text
		Angles.append(ang)
		
	return Angles

def annot_classes(annotation_file):
	annot_file = open(annotation_file, 'r')
	tree = ET.parse(annot_file)
	root = tree.getroot()
	Classes=[annotation_file]
	objs = root.find('HRSC_Objects').findall('HRSC_Object')
	for j in objs:
		Class_id = j.find('Class_ID').text
		Classes.append(Class_id)
		
	return Classes

def file_OBB_reader(annotation_file, dataset_type):
	annot_file = open(annotation_file, 'r')
	objs_ex = False
	extra_desc=[]
	if dataset_type=='HRSC2016':
		tree = ET.parse(annot_file)
		root = tree.getroot()
		objs = root.find('HRSC_Objects')
		n_obj=0
		for j in objs.iter('HRSC_Object'):
			Class_id = j.find('Class_ID').text
			x_min = int(j.find('box_xmin').text)
			y_min = int(j.find('box_ymin').text)
			x_max = int(j.find('box_xmax').text)
			y_max = int(j.find('box_ymax').text)
			cx = (x_min + x_max)/2
			cy = (y_min + y_max)/2
			w = x_max - x_min
			h = y_max - y_min
			if (n_obj==0):
				coors = np.array([Class_id, cx, cy, w, h], dtype=object)
			else:
				coors = np.vstack([coors, [Class_id, cx, cy, w, h]])
			n_obj+=1
			objs_ex=True
	elif dataset_type=='ShipRSImageNet':
		tree = ET.parse(annot_file)
		root = tree.getroot()
		n_obj=0
		for j in root.iter('object'):
			Class_id = j.find('level_3').text
			bndbox = j.find('bndbox')
			x_min = int(bndbox.find('xmin').text)
			y_min = int(bndbox.find('ymin').text)
			x_max = int(bndbox.find('xmax').text)
			y_max = int(bndbox.find('ymax').text)
			cx = (x_min + x_max)/2
			cy = (y_min + y_max)/2
			w = x_max - x_min
			h = y_max - y_min
			if (n_obj==0):
				coors = np.array([Class_id, cx, cy, w, h], dtype=object)
			else:
				coors = np.vstack([coors, [Class_id, cx, cy, w, h]])
			n_obj+=1
			objs_ex=True
	elif dataset_type=='DOTA':
		n_obj=0
		lines_hbb = annot_file.readlines()
		difs=[]
		for j, line_hbb in enumerate(lines_hbb):
			if j==0:
				source = line_hbb
			elif j==1:
				gsd = line_hbb
			else:
				coors_hbb = np.array(line_hbb.split(" ")[:8]).astype(float)
				cl = line_hbb.split(" ")[8]
				dif = line_hbb.split(" ")[9]
				Class_id = np.where(DOTA_cls==cl)[0]+1
				x_min = coors_hbb[0::2].min()
				y_min = coors_hbb[1::2].min()
				x_max = coors_hbb[0::2].max()
				y_max = coors_hbb[1::2].max()
				cx = (x_min + x_max)/2
				cy = (y_min + y_max)/2
				w = x_max - x_min
				h = y_max - y_min
				if (n_obj==0):
					coors = np.array([Class_id, cx, cy, w, h])
				else:
					coors = np.vstack([coors, [Class_id, cx, cy, w, h]])
				difs.append(dif)
				n_obj+=1
				objs_ex=True
		difs = np.array(difs)
		extra_desc = [source, gsd, difs]

	annot_file.close()
	if objs_ex==True:
		return coors.reshape(-1,5), objs_ex, extra_desc
	else:
		return None, objs_ex, extra_desc

def create_files(annotation_file, im_w, im_h, objs_rboxes, objs_hboxes, desc, Classes, dataset_type):
	if dataset_type=='HRSC2016':
		root = ET.Element('HRSC_Image')
		Img_ID = ET.SubElement(root, 'Img_ID')
		Fmt = ET.SubElement(root, 'Img_FileFmt')
		img_width = ET.SubElement(root, 'Img_SizeWidth')
		img_height = ET.SubElement(root, 'Img_SizeHeight')
		Img_ID.text = annotation_file.split('/')[-1][:-4]
		Fmt.text = 'bmp'
		img_width.text = str(im_w)
		img_height.text = str(im_h)
		Objects = ET.SubElement(root, 'HRSC_Objects')
		for i, obj in enumerate(objs_rboxes):
			Obj = ET.SubElement(Objects, 'HRSC_Object')

			cl_id = ET.SubElement(Obj, 'Class_ID')
			cl_id.text = str(Classes[i])

			xmin = ET.SubElement(Obj, 'box_xmin')
			ymin = ET.SubElement(Obj, 'box_ymin')
			xmax = ET.SubElement(Obj, 'box_xmax')
			ymax = ET.SubElement(Obj, 'box_ymax')
			xmin.text = str(objs_hboxes[i][0])
			ymin.text = str(objs_hboxes[i][1])
			xmax.text = str(objs_hboxes[i][2])
			ymax.text = str(objs_hboxes[i][3])

			obj_box = objs_rboxes[i]
			side1 = np.sqrt(((obj_box[1]-obj_box[0])**2).sum())
			side2 = np.sqrt(((obj_box[2]-obj_box[1])**2).sum())
			side3 = np.sqrt(((obj_box[3]-obj_box[2])**2).sum())
			side4 = np.sqrt(((obj_box[0]-obj_box[3])**2).sum())

			if side1>=side2:
				w1 = side1
				h1 = side2
				w2 = side3
				h2 = side4
			else:
				w1 = side2
				h1 = side1
				w2 = side4
				h2 = side3

			y1 = obj_box[np.where(obj_box[:,1]==np.partition(obj_box[:,1], -2)[-2])][0][1]
			y2 = obj_box[np.where(obj_box[:,1]==obj_box[:,1].min())][0][1]
			x1 = obj_box[np.where(obj_box[:,1]==np.partition(obj_box[:,1], -2)[-2])][0][0]
			x2 = obj_box[np.where(obj_box[:,1]==obj_box[:,1].min())][0][0]

			sid = np.sqrt((y1-y2)**2+(x1-x2)**2)

			if (np.abs(sid-w1)<np.abs(sid-h1)):
				orient = np.arctan((y1-y2)/(x1-x2))
			elif ((np.abs(sid-w1)>np.abs(sid-h1)) and ((y1-y2)/(x1-x2)>0)):
				orient = np.arctan((y1-y2)/(x1-x2)) - np.pi/2
			elif ((np.abs(sid-w1)>np.abs(sid-h1)) and ((y1-y2)/(x1-x2)<0)):
				orient = np.arctan((y1-y2)/(x1-x2)) + np.pi/2
			else:
				print(np.sqrt((y1-y2)**2+(x1-x2)**2), (y1-y2)/(x1-x2), w1, h1, w2, h2)
				print('An unexpected scenario occurred. Continuing...')
				continue

			cx = ET.SubElement(Obj, 'mbox_cx')
			cy = ET.SubElement(Obj, 'mbox_cy')
			w = ET.SubElement(Obj, 'mbox_w')
			h = ET.SubElement(Obj, 'mbox_h')
			ang = ET.SubElement(Obj, 'mbox_ang')
			cx.text = str((obj_box[:,0].max() + obj_box[:,0].min())/2)
			cy.text = str((obj_box[:,1].max() + obj_box[:,1].min())/2)
			w.text = str(w1)
			h.text = str(h1)
			ang.text = str(orient)
		
		xml_string = ET.tostring(root, encoding='utf-8', method='xml')
		formatted_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")
		with open(annotation_file, 'w') as xml_file:
			xml_file.write(formatted_xml)

	elif dataset_type=='ShipRSImageNet':
		root = ET.Element('annotation')
		filename = ET.SubElement(root, 'filename')
		filename.text = annotation_file.split('/')[-1][:-4]
		size = ET.SubElement(root, 'size')
		img_width = ET.SubElement(size, 'width')
		img_height = ET.SubElement(size, 'height')
		img_depth = ET.SubElement(size, 'depth')
		img_width.text = str(im_w)
		img_height.text = str(im_h)
		img_depth.text = str(3)
		for i, obj in enumerate(objs_rboxes):
			Obj = ET.SubElement(root, 'object')
			Cl_l0 = ET.SubElement(Obj, 'level_0')
			Cl_l3 = ET.SubElement(Obj, 'level_3')
			polygon = ET.SubElement(Obj, 'polygon')
			bndbox = ET.SubElement(Obj, 'bndbox')

			Cl_l3.text=str(Classes[i])
			if Classes[i]==50:
				Cl_l0.text = '2'
			else:
				Cl_l0.text = '1'

			xmin = ET.SubElement(bndbox, 'xmin')
			ymin = ET.SubElement(bndbox, 'ymin')
			xmax = ET.SubElement(bndbox, 'xmax')
			ymax = ET.SubElement(bndbox, 'ymax')
			xmin.text = str(objs_hboxes[i][0])
			ymin.text = str(objs_hboxes[i][1])
			xmax.text = str(objs_hboxes[i][2])
			ymax.text = str(objs_hboxes[i][3])

			x1 = ET.SubElement(polygon, 'x1')
			y1 = ET.SubElement(polygon, 'y1')
			x2 = ET.SubElement(polygon, 'x2')
			y2 = ET.SubElement(polygon, 'y2')
			x3 = ET.SubElement(polygon, 'x3')
			y3 = ET.SubElement(polygon, 'y3')
			x4 = ET.SubElement(polygon, 'x4')
			y4 = ET.SubElement(polygon, 'y4')

			x1.text = str(objs_rboxes[i][0][0])
			y1.text = str(objs_rboxes[i][0][1])
			x2.text = str(objs_rboxes[i][1][0])
			y2.text = str(objs_rboxes[i][1][1])
			x3.text = str(objs_rboxes[i][2][0])
			y3.text = str(objs_rboxes[i][2][1])
			x4.text = str(objs_rboxes[i][3][0])
			y4.text = str(objs_rboxes[i][3][1])
		
		xml_string = ET.tostring(root, encoding='utf-8', method='xml')
		formatted_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")
		with open(annotation_file, 'w') as xml_file:
			xml_file.write(formatted_xml)

	elif dataset_type=='DOTA':
		f = open(annotation_file, 'w')
		f.writelines(desc[0])
		f.writelines(desc[1])
		for i, obj in enumerate(objs_rboxes):
			x1 = str(objs_rboxes[i][0][0])
			y1 = str(objs_rboxes[i][0][1])
			x2 = str(objs_rboxes[i][1][0])
			y2 = str(objs_rboxes[i][1][1])
			x3 = str(objs_rboxes[i][2][0])
			y3 = str(objs_rboxes[i][2][1])
			x4 = str(objs_rboxes[i][3][0])
			y4 = str(objs_rboxes[i][3][1])

			cl = DOTA_cls[Classes[i]-1]
			dif = str(desc[2][i])
			f.writelines(x1+' '+y1+' '+x2+' '+y2+' '+x3+' '+y3+' '+x4+' '+y4+' '+cl+' '+dif)
		f.close()

		"""
		root = ET.Element('annotation')
		filename = ET.SubElement(root, 'filename')
		filename.text = annotation_file.split('/')[-1][:-4]+'.png'

		size = ET.SubElement(root, 'size')
		img_width = ET.SubElement(size, 'width')
		img_height = ET.SubElement(size, 'height')
		img_depth = ET.SubElement(size, 'depth')
		img_width.text = str(im_w)
		img_height.text = str(im_h)
		img_depth.text = str(3)
		
		for i, obj in enumerate(objs_rboxes):
			Obj = ET.SubElement(root, 'object')

			Cl = ET.SubElement(Obj, 'Class')
			Cl.text = DOTA_cls[Classes[i]-1]

			Area = ET.SubElement(Obj, 'area')
			area = np.sqrt((objs_rboxes[i][1][0]-objs_rboxes[i][0][0])**2+(objs_rboxes[i][1][1]-objs_rboxes[i][0][1])**2)*np.sqrt((objs_rboxes[i][2][0]-objs_rboxes[i][1][0])**2+(objs_rboxes[i][2][1]-objs_rboxes[i][1][1])**2)
			Area.text = str(int(area))

			#diff = ET.SubElement(Obj, 'difficult')
			#diff.text = str(int(difs[k]))

			bndbox = ET.SubElement(Obj, 'bndbox')
			xmin = ET.SubElement(bndbox, 'xmin')
			ymin = ET.SubElement(bndbox, 'ymin')
			xmax = ET.SubElement(bndbox, 'xmax')
			ymax = ET.SubElement(bndbox, 'ymax')
			xmin.text = str(objs_hboxes[i][0])
			ymin.text = str(objs_hboxes[i][1])
			xmax.text = str(objs_hboxes[i][2])
			ymax.text = str(objs_hboxes[i][3])

			polygon = ET.SubElement(Obj, 'polygon')
			x1 = ET.SubElement(polygon, 'x1')
			y1 = ET.SubElement(polygon, 'y1')
			x2 = ET.SubElement(polygon, 'x2')
			y2 = ET.SubElement(polygon, 'y2')
			x3 = ET.SubElement(polygon, 'x3')
			y3 = ET.SubElement(polygon, 'y3')
			x4 = ET.SubElement(polygon, 'x4')
			y4 = ET.SubElement(polygon, 'y4')

			x1.text = str(objs_rboxes[i][0][0])
			y1.text = str(objs_rboxes[i][0][1])
			x2.text = str(objs_rboxes[i][1][0])
			y2.text = str(objs_rboxes[i][1][1])
			x3.text = str(objs_rboxes[i][2][0])
			y3.text = str(objs_rboxes[i][2][1])
			x4.text = str(objs_rboxes[i][3][0])
			y4.text = str(objs_rboxes[i][3][1])
		"""


	
