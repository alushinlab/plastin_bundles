#!/home/alus_soft/matt_EMAN2/bin/python
####################################################################################################
# imports
import argparse; import sys
import matplotlib.pyplot as plt
import numpy as np
import mrcfile 
from EMAN2 import *
####################################################################################################
parser = argparse.ArgumentParser('Generates either a soft or sharp circular mask of the specified box size and ring size.')
parser.add_argument('--boxsize', type=int, help='dimensions of box size for circular mask')
parser.add_argument('--diam', type=int, help='diameter of soft circular mask')
parser.add_argument('--width', type=int, help='1/e width of Gaussian falloff (both inner and outer) in pixels. default=4')
parser.add_argument('--sharp', type=str, help='enter 1 for this flag if you would like a sharp mask instead of a soft mask.')

args = parser.parse_args()
print('')
if(args.boxsize == None or args.diam == None):
	print('You must specify a boxsize AND a diameter')
	sys.exit('Exiting...')

if(args.sharp == None):
	isSoft = True
else:
	isSoft = False

if(args.width == None and isSoft):
	print('No falloff width specified, using default of 4')
	width = 4
elif(isSoft):
	width = args.width

boxsize = args.boxsize
diam = args.diam

if(isSoft):
	output_file_name = 'soft_mask_%s_%s_%s.mrc'%(boxsize,diam,width)
else:
	output_file_name = 'sharp_mask_%s_%s.mrc'%(boxsize,diam)

ones = np.ones((boxsize,boxsize))
ones_EM = EMNumPy.numpy2em(ones)
if(isSoft):
	soft_mask = ones_EM.process('mask.soft', {'outer_radius':(diam)/2.0, 'width':width})
else:
	soft_mask = ones_EM.process('mask.sharp', {'outer_radius':(diam)/2.0})

with mrcfile.new(output_file_name, overwrite=True) as mrc:
	mrc.set_data(EMNumPy.em2numpy(soft_mask).astype('float32'))

print('Written output file as : ' + output_file_name)



