import xml.etree.ElementTree as ET
import numpy as np 
import matplotlib.pyplot as plt 
import os

ShipRSImageNet_classes_dict = {'100000001':0, '100000002':1, '100000003':2, '100000004':3, '100000005':4, 
					 		   '100000006':5, '100000007':6, '100000008':7, '100000009':8, '100000010':9,
					 		   '100000011':10, '100000012':11, '100000013':12, '100000014':13, '100000015':14,
					 		   '100000016':15, '100000017':16, '100000018':17, '100000019':18, '100000020':19,
					 		   '100000022':20, '100000024':21, '100000025':22, '100000026':23, '100000027':24,
					 	   	   '100000028':25, '100000029':26, '100000030':27, '100000031':28, '100000032':29,
					 		   '100000033':30}

ShipRSImageNet_CLASSES_NAMES = {0:'Other Ship', 1:'Other Warship', 2:'Submarine', 3:'Other Aircraft Carrier',
                    4:'Enterprise', 5:'Nimitz', 6:'Midway', 7:'Ticonderoga',
                    8:'Other Destroyer', 9:'Atago DD', 10:'Arleigh Burke DD', 11:'Hatsuyuki DD',
                    12:'Hyuga DD', 13:'Asagiri DD', 14:'Other Frigate', 15:'Perry FF', 16:'Patrol',
                    17:'Other Landing', 18:'YuTing LL', 19:'YuDeng LL',
                    20:'YuDao LL', 21:'YuZhao LL', 22:'Austin LL', 23:'Osumi LL',
                    24:'Wasp LL', 25:'LSD 41 LL', 26:'LHA LL', 27:'Commander',
                    28:'Other Auxiliary Ship', 29:'Medical Ship', 30:'Test Ship',
                    31:'Training Ship', 32:'AOE', 33:'Masyuu AS', 34:'Sanantonio AS',
                    35:'EPF', 36:'Other Merchant', 37:'Container Ship', 38:'RoRo',
                    39:'Cargo', 40:'Barge', 41:'Tugboat', 42:'Ferry', 43:'Yacht',
                    44:'Sailboat', 45:'Fishing Vessel', 46:'Oil Tanker', 47:'Hovercraft',
                    48:'Motorboat', 49:'Dock'}		 

path_to_aug = # Path to annotations directory
classes=[]
angles=[]
bin_granularity=10
for f in np.array(sorted(os.listdir(path_to_aug))):
	#if f[-5]=='1':
		tree = ET.parse(os.path.join(path_to_aug, f))
		root = tree.getroot()
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

			xs = np.array([x1,x2,x3,x4])
			ys = np.array([y1,y2,y3,y4])

			side01 = np.sqrt((xs[1]-xs[0])**2+(ys[1]-ys[0])**2)
			side12 = np.sqrt((xs[2]-xs[1])**2+(ys[2]-ys[1])**2)

			ymin = ys.min()
			xmax = xs.max()
			x_ymin = xs[np.where(ys==ymin)]
			y_xmax = ys[np.where(xs==xmax)]
			#print(xmax, ymin, x_ymin, y_xmax)
			if len(y_xmax)>1:
				y_xmax = y_xmax[np.where(y_xmax!=ymin)]
				if len(y_xmax)>1:
					y_xmax = y_xmax[np.where(np.abs(y_xmax-ymin)==np.abs(y_xmax-ymin).min())]
			if len(x_ymin)>1:
				#print(f'OH OH, {len}')
				x_ymin = x_ymin[np.where(x_ymin!=xmax)]
				if len(x_ymin)>1:
					x_ymin = x_ymin[np.where(np.abs(x_ymin-xmax)==np.abs(x_ymin-xmax).min())]
			side1 = np.sqrt((y_xmax-ymin)**2+(xmax-x_ymin)**2)
			#print(side1)
			if side1==side01:
				side2 = side12
			else:
				side2 = side01

			if side1>=side2:
				angle = np.pi/2 + np.arctan2(y_xmax-ymin, xmax-x_ymin)
			else:
				angle = np.arctan2(y_xmax-ymin, xmax-x_ymin)

			if ((angle*180/np.pi)%bin_granularity)>(bin_granularity/2):
				round_angle = (angle*180/np.pi)//bin_granularity*bin_granularity+bin_granularity
			else:
				round_angle = (angle*180/np.pi)//bin_granularity*bin_granularity
			classes.append(int(obj.find('level_3').text)-1)
			angles.append(round_angle)
			#angles.append(float(obj.find('mbox_ang').text)*180/np.pi + 90)
			#print(float(obj.find('mbox_ang').text)*180/np.pi)


classes = np.array(classes)
angles = np.array(angles)
angles[np.where(angles==180)]=0

plt.figure(figsize=(5,5))
plt.hist(angles, bins=18)
plt.xticks(np.linspace(0,180,19))
plt.xlabel('Orientations (deg)')
plt.ylabel('# Objects')
plt.show()

"""
for cl in np.unique(classes):
	angles_cl = angles[np.where(classes==cl)]
	plt.figure(figsize=(5,5))
	plt.hist(angles_cl, bins=18)
	plt.xlabel('Orientations (deg)')
	plt.ylabel('# Objects')
	plt.title(f'{ShipRSImageNet_CLASSES_NAMES[cl]} class angle histogram')
	#plt.savefig(f'angle_histograms_per_class_aug/{HRSC_CLASSES_NAMES[cl]}_hist_2.jpg')
	plt.show()

"""

