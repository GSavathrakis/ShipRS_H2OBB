import xml.etree.ElementTree as ET
import numpy as np 
import matplotlib.pyplot as plt 
import os

HRSC_classes_dict = {'100000001':0, '100000002':1, '100000003':2, '100000004':3, '100000005':4, 
					 '100000006':5, '100000007':6, '100000008':7, '100000009':8, '100000010':9,
					 '100000011':10, '100000012':11, '100000013':12, '100000014':13, '100000015':14,
					 '100000016':15, '100000017':16, '100000018':17, '100000019':18, '100000020':19,
					 '100000022':20, '100000024':21, '100000025':22, '100000026':23, '100000027':24,
					 '100000028':25, '100000029':26, '100000030':27, '100000031':28, '100000032':29,
					 '100000033':30}

HRSC_CLASSES_NAMES = {0:'ship', 1:'aircraft carrier', 2:'warcraft', 3:'merchant ship',
                    4:'Nimitz', 5:'Enterprise', 6:'Arleigh Burke', 7:'WhidbeyIsland',
                    8:'Perry', 9:'Sanantonio', 10:'Ticonderoga', 11:'Kitty Hawk',
                    12:'Kuznetsov', 13:'Abukuma', 14:'Austen', 15:'Tarawa', 16:'Blue Ridge',
                    17:'Container', 18:'OXo|--)', 19:'Car carrier([]==[])',
                    20:'Hovercraft', 21:'yacht', 22:'CntShip(_|.--.--|_]=', 23:'Cruise',
                    24:'submarine', 25:'lute', 26:'Medical', 27:'Car carrier(======|',
                    28:'Ford-class', 29:'Midway-class', 30:'Invincible-class'}		 

path_to_aug = # Path to annotations directory
classes=[]
angles=[]
bin_granularity=10
for f in np.array(sorted(os.listdir(path_to_aug))):
	#if f[-5]=='1':
		tree = ET.parse(os.path.join(path_to_aug, f))
		root = tree.getroot()
		objs = root.find('HRSC_Objects').findall('HRSC_Object')
		for obj in objs:
			if ((float(obj.find('mbox_ang').text)*180/np.pi+90)%bin_granularity)>(bin_granularity/2):
				round_angle = (float(obj.find('mbox_ang').text)*180/np.pi+90)//bin_granularity*bin_granularity+bin_granularity
			else:
				round_angle = (float(obj.find('mbox_ang').text)*180/np.pi+90)//bin_granularity*bin_granularity
			classes.append(HRSC_classes_dict[obj.find('Class_ID').text])
			angles.append(round_angle)


classes = np.array(classes)
angles = np.array(angles)
angles[np.where(angles==180)]=0
n_objs_per_or=[]
for i in range(0,18):
	n_objs_per_or.append(len(angles[np.where(angles==i*10)]))
n_objs_per_or = np.array(n_objs_per_or)





plt.figure(figsize=(12,12))
plt.vlines(np.linspace(0,170,18), 0, n_objs_per_or, linewidth=35)
plt.xticks(np.linspace(0,170,18), fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Orientations (deg)', fontsize=24)
plt.ylabel('# Objects', fontsize=24)
plt.show()

"""
for cl in np.unique(classes):
	angles_cl = angles[np.where(classes==cl)]
	plt.figure(figsize=(5,5))
	plt.hist(angles_cl, bins=18)
	plt.xlabel('Orientations (deg)')
	plt.ylabel('# Objects')
	plt.title(f'{HRSC_CLASSES_NAMES[cl]} class angle histogram')
	#plt.savefig(f'angle_histograms_per_class_aug/{HRSC_CLASSES_NAMES[cl]}_hist_2.jpg')
	plt.show()
"""

