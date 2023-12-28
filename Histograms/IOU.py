import numpy as np 
import matplotlib.pyplot as plt 

IOUs = np.load('../OBB_generation/IOU.npy')


plt.figure(figsize=(4,4))
plt.hist(IOUs, bins=40)
plt.xlabel('IOU')
plt.ylabel('# Objects')
plt.show()

print(f'% objects with IOU>90%: {len(np.where(IOUs>=0.9)[0])/len(IOUs)*100}')
print(f'% objects with IOU>80%: {len(np.where(IOUs>=0.8)[0])/len(IOUs)*100}')
print(f'% objects with IOU>70%: {len(np.where(IOUs>=0.7)[0])/len(IOUs)*100}')
print(f'% objects with IOU>60%: {len(np.where(IOUs>=0.6)[0])/len(IOUs)*100}')

