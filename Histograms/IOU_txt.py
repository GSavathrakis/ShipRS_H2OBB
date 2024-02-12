import numpy as np 
import matplotlib.pyplot as plt 
import argparse

parser = argparse.ArgumentParser('Select IoU file', add_help=False)
parser.add_argument('--IoU_file', type=str)
args = parser.parse_args() 


def main(args):
    IOUs = np.loadtxt(args.IoU_file, dtype=float)


    plt.figure(figsize=(5,5))
    plt.hist(IOUs, bins=40)
    plt.xlabel('IOU')
    plt.savefig('IoU_DOTA')
    plt.close()

    print(f'% objects with IOU>90%: {len(np.where(IOUs>=0.9)[0])/len(IOUs)*100}')
    print(f'% objects with IOU>80%: {len(np.where(IOUs>=0.8)[0])/len(IOUs)*100}')
    print(f'% objects with IOU>70%: {len(np.where(IOUs>=0.7)[0])/len(IOUs)*100}')
    print(f'% objects with IOU>60%: {len(np.where(IOUs>=0.6)[0])/len(IOUs)*100}')

if __name__ == '__main__':
	main(args)