import os
import cv2
from tqdm import tqdm
import numpy as np 
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse

parser = argparse.ArgumentParser('Transform DOTA txt to VOC format', add_help=False)
parser.add_argument('--hbb_dir_name', type=str)
parser.add_argument('--obb_dir_name', type=str)
parser.add_argument('--img_dir_name', type=str)
args = parser.parse_args() 

def main(args)
	hbb_dir_name = args.hbb_dir_name
	obb_dir_name = args.obb_dir_name
	img_dir_name = args.img_dir_name

	hbb_dir = sorted(np.array(os.listdir(hbb_dir_name)))
	obb_dir = sorted(np.array(os.listdir(obb_dir_name)))
	img_dir = sorted(np.array(os.listdir(img_dir_name)))

	if not os.path.exists('VOC_Format'):
		os.mkdir('VOC_Format')

	assert len(hbb_dir)==len(obb_dir)

	for i in tqdm(range(len(hbb_dir)), desc='Transformed annotations'):
		fil_i_hbb = hbb_dir[i]
		fil_i_obb = obb_dir[i]

		assert fil_i_obb == fil_i_hbb
		assert fil_i_hbb[:-4] == img_dir[i][:-4]

		file_path_hbb = os.path.join(hbb_dir_name, fil_i_hbb)
		file_path_obb = os.path.join(obb_dir_name, fil_i_obb)
		img_path = os.path.join(img_dir_name, img_dir[i])

		img = cv2.imread(img_path)
		(H, W, d) = img.shape
		
		fil_hbb = open(file_path_hbb, 'r')
		lines_hbb = fil_hbb.readlines()

		fil_obb = open(file_path_obb, 'r')
		lines_obb = fil_obb.readlines()

		coors_obb_all=[]
		coors_hbb_all=[]
		clss=[]
		areas=[]
		difs=[]
		for j, line_hbb in enumerate(lines_hbb):
			if j>1:
				coors_hbb = np.array(line_hbb.split(" ")[:8]).astype(float)
				coors_obb = np.array(lines_obb[j].split(" ")[:8]).astype(float)
				cl = line_hbb.split(" ")[8]
				dif = line_hbb.split(" ")[9]
				area = np.sqrt((coors_obb[2]-coors_obb[0])**2+(coors_obb[3]-coors_obb[1])**2)*np.sqrt((coors_obb[4]-coors_obb[2])**2+(coors_obb[5]-coors_obb[3])**2)

				coors_hbb_all.append(coors_hbb)
				coors_obb_all.append(coors_obb)
				clss.append(cl)
				difs.append(dif)
				areas.append(area)

		root = ET.Element('annotation')
		filename = ET.SubElement(root, 'filename')
		filename.text = img_dir[i]

		size = ET.SubElement(root, 'size')
		img_width = ET.SubElement(size, 'width')
		img_height = ET.SubElement(size, 'height')
		img_depth = ET.SubElement(size, 'depth')
		img_width.text = str(W)
		img_height.text = str(H)
		img_depth.text = str(d)
		
		for k, obj in enumerate(coors_obb_all):
			Obj = ET.SubElement(root, 'object')

			Cl = ET.SubElement(Obj, 'Class')
			Cl.text = clss[k]

			Area = ET.SubElement(Obj, 'area')
			Area.text = str(int(areas[k]))

			diff = ET.SubElement(Obj, 'difficult')
			diff.text = str(int(difs[k]))

			bndbox = ET.SubElement(Obj, 'bndbox')
			xmin = ET.SubElement(bndbox, 'xmin')
			ymin = ET.SubElement(bndbox, 'ymin')
			xmax = ET.SubElement(bndbox, 'xmax')
			ymax = ET.SubElement(bndbox, 'ymax')
			xmin.text = str(int(coors_hbb_all[k][0::2].min()))
			ymin.text = str(int(coors_hbb_all[k][1::2].min()))
			xmax.text = str(int(coors_hbb_all[k][0::2].max()))
			ymax.text = str(int(coors_hbb_all[k][1::2].max()))

			polygon = ET.SubElement(Obj, 'polygon')
			x1 = ET.SubElement(polygon, 'x1')
			y1 = ET.SubElement(polygon, 'y1')
			x2 = ET.SubElement(polygon, 'x2')
			y2 = ET.SubElement(polygon, 'y2')
			x3 = ET.SubElement(polygon, 'x3')
			y3 = ET.SubElement(polygon, 'y3')
			x4 = ET.SubElement(polygon, 'x4')
			y4 = ET.SubElement(polygon, 'y4')

			x1.text = str(int(coors_obb_all[k][0]))
			y1.text = str(int(coors_obb_all[k][1]))
			x2.text = str(int(coors_obb_all[k][2]))
			y2.text = str(int(coors_obb_all[k][3]))
			x3.text = str(int(coors_obb_all[k][4]))
			y3.text = str(int(coors_obb_all[k][5]))
			x4.text = str(int(coors_obb_all[k][6]))
			y4.text = str(int(coors_obb_all[k][7]))
		

		annotation_file_name = fil_i_hbb[:-4]+'.xml'
		xml_string = ET.tostring(root, encoding='utf-8', method='xml')
		formatted_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")
		with open(os.path.join('VOC_Format', annotation_file_name), 'w') as xml_file:
			xml_file.write(formatted_xml)

if __name__ == '__main__':
	main(args)
