#!/home/greg/.conda/envs/matt_TF/bin/python
################################################################################
print('Loading python packages...')
################################################################################
# import of python packages
import numpy as np
import matplotlib.pyplot as plt
import keras
import mrcfile
import random
from tqdm import tqdm
from keras import layers
from keras.models import Model
import tensorflow as tf
#from EMAN2 import *
from scipy import interpolate; from scipy.ndimage import filters
from skimage.morphology import skeletonize_3d; import scipy
import keras.backend as K
import glob
import os
from scipy.ndimage.measurements import center_of_mass
################################################################################
print('Python packages loaded. Setting CUDA environment...')
GPU_IDX = 2
os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_IDX)
################################################################################
################# Functions for opterating on whole micrographs ################
def slice_up_micrograph(real_data, increment, box_size,hist_matcher):#, box_size):
	extractions = []
	for i in range(0, real_data.shape[0]-box_size, increment):
		for j in range(0, real_data.shape[1]-box_size, increment):
			extraction = -1.0*real_data[i:i+box_size,j:j+box_size]
			extraction = hist_match(extraction, hist_matcher)
			extractions.append(extraction)
	extractions = np.moveaxis(np.asarray(extractions), 0,-1)
	extractions = (extractions - np.mean(extractions, axis=(0,1))) / np.std(extractions, axis=(0,1))
	return np.moveaxis(extractions, -1, 0)

def stitch_back_seg(shape, preds, increment, box_size):
	stitch_back = np.zeros((shape))
	cntr=0
	for i in range(0, stitch_back.shape[0]-box_size, increment):
		for j in range(0, stitch_back.shape[1]-box_size, increment):
			stitch_back[i:i+box_size, j:j+box_size] = np.max(np.stack((preds[cntr], stitch_back[i:i+box_size, j:j+box_size])),axis=0)
			cntr=cntr+1
	return stitch_back

def hist_match(source, template):
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()
	# get the set of unique pixel values and their corresponding indices and counts
	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)
	
	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]
	
	# interpolate linearly to find the pixel values in the template image
	# that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
	return interp_t_values[bin_idx].reshape(oldshape)

def implement_NMS(semMap):
	################################################################################
	# Four parameters
	box_size = 96
	half_box_size = int(box_size / 2)
	occ_high = 0.5
	occ_low = 0.10 # 0.15
	radius = 36
	inner_box = 12
	inner_box_half = int(inner_box / 2)
	occ_smallBox = 0.80
	################################################################################
	picks = np.argwhere(semMap==1)
	picks_noborder = [] # make more efficient with np.where
	for x in range(0,picks.shape[0]):
		if picks[x][0] >= half_box_size and picks[x][1] >= half_box_size and picks[x][0] < semMap.shape[0]-half_box_size and picks[x][1] < semMap.shape[1]-half_box_size:
			picks_noborder.append(picks[x])
	
	picks_noborder = np.asarray(picks_noborder)
	
	picks_occupancy = []
	box_area = float(box_size**2)
	for x in (range(0,len(picks_noborder))):
		box = semMap[picks_noborder[x][0]-half_box_size:picks_noborder[x][0]+half_box_size,picks_noborder[x][1]-half_box_size:picks_noborder[x][1]+half_box_size]
		occupancy = np.sum(box) / box_area
		if occupancy > occ_low and occupancy < occ_high:
			picks_occupancy.append(picks_noborder[x])
	
	picks_occupancy = np.asarray(picks_occupancy)
	################################################################################
	picks_occupancy_2 = []
	smallBox_area = float(inner_box**2)
	for x in (range(0,len(picks_occupancy))):
		box = semMap[picks_occupancy[x][0]-inner_box_half:picks_occupancy[x][0]+inner_box_half,picks_occupancy[x][1]-inner_box_half:picks_occupancy[x][1]+inner_box_half]
		occupancy = np.sum(box) / smallBox_area
		if occupancy > occ_smallBox:
			picks_occupancy_2.append(picks_occupancy[x])
	
	picks_occupancy_2 = np.asarray(picks_occupancy_2)
	picks_occupancy_2 = picks_occupancy_2
	
	pick_com = []
	for x in (range(0, len(picks_occupancy_2))):
		box = semMap[picks_occupancy_2[x][0]-half_box_size:picks_occupancy_2[x][0]+half_box_size,picks_occupancy_2[x][1]-half_box_size:picks_occupancy_2[x][1]+half_box_size]
		com = center_of_mass(box)
		if(np.abs(com[0]-half_box_size) < 3 or np.abs(com[1]-half_box_size) < 3):
			pick_com.append(picks_occupancy_2[x])
	
	pick_com = np.asarray(pick_com)
	semMap_filtered = semMap.copy()
	for i in range(0, len(pick_com)):
		semMap_filtered[pick_com[i][0],pick_com[i][1]] = 2
	
	semMap_filtered = semMap_filtered-1
	semMap_filtered[semMap_filtered == -1] = 0
	semMap_skel = skeletonize_3d(semMap_filtered)
	picks_skel = np.argwhere(semMap_skel==255)
	
	# implement NMS algorithm
	nms = picks_skel.copy()[::8]
	project_dist_matrix=1
	if(len(nms) == 0):
		return []
	while(np.sum(project_dist_matrix) != 0):
		sq_len = len(nms)
		coord1 = np.repeat(np.expand_dims(nms, axis=0),  len(nms), axis=0).reshape(sq_len,sq_len,2)
		coord2 = np.repeat(np.expand_dims(nms, axis=-1), len(nms), axis=0).reshape(sq_len,sq_len,2)
		dist_matrix = np.linalg.norm(coord1 - coord2, axis=-1)
		neighbor_matrix = dist_matrix < radius # cutoff dist for being called a neighbor
		dist_matrix_binarized = np.multiply(-1*np.eye(len(dist_matrix))+1,neighbor_matrix)
		project_dist_matrix = np.sum(dist_matrix_binarized,axis=0)
		val_to_pop = np.argmax(project_dist_matrix)
		nms = np.delete(nms, val_to_pop, axis=0)
	
	return nms

def starify(*args):
	return (''.join((('%.6f'%i).rjust(13))  if not isinstance(i,int) else ('%d'%i).rjust(13) for i in args) + ' \n')[1:]

################################################################################
# load trained Fully Convolutional Network for semantic segmentation
################################################################################
print('Loading neural network models'); print('')
model_path = './semSeg_50ktrain_catCrossEnt.h5'
FCN = keras.models.load_model(model_path)

print('Network loaded')
# Load one test image for histogram matching
hist_match_dir = './'
with mrcfile.open(hist_match_dir + 'actin_rotated%05d.mrc'%1) as mrc:
	hist_matcher = mrc.data

################################################################################
# Load real micrographs
file_names = sorted(glob.glob('../Micrographs_bin4/*_bin4.mrc'))
import random 
random.seed(4)
file_names = random.sample(file_names, 100)
img_num = len(file_names)#len(sorted(glob.glob('../Micrographs_bin4/*_bin4.mrc')))
img_start = (img_num / 4)*GPU_IDX
img_end = (img_num / 4)*(GPU_IDX+1)
file_names = file_names[img_start:img_end]

for i in tqdm(range(0, len(file_names))):
	big_micrograph_name = file_names[i]
	with mrcfile.open(big_micrograph_name) as mrc:
		real_data = mrc.data
	
	################################################################################
	# Divide up the whole micrograph to feed to the FCN for semantic segmentation
	increment = 48
	extractions = slice_up_micrograph(real_data, increment, 192, hist_matcher)
	preds = FCN.predict(np.expand_dims(0.9*extractions, axis=-1))
	stitch_back = stitch_back_seg(real_data.shape, preds[:,:,:,2], increment, 192)
	with mrcfile.new('./pngs_semSeg/'+big_micrograph_name[:-4].split('/')[-1]+'.mrc', overwrite=True) as mrc:
		mrc.set_data((stitch_back>0.85).astype('float32'))
	
	binarized_stitch_back = (stitch_back > 0.85).astype('float32')
	picks = implement_NMS(binarized_stitch_back)
	
	_=plt.imshow(real_data, origin='lower', cmap=plt.cm.gray)
	if(picks != []): _=plt.scatter(picks[:,1], picks[:,0], s=8)
	_=plt.tight_layout()
	_=plt.savefig('pngs/'+big_micrograph_name[:-4].split('/')[-1]+'.png', dpi=600)
	_=plt.clf()
	
	################################################################################
	# Prepare star file
	header = '# RELION; version 3.0\n\ndata_\n\nloop_ \n_rlnCoordinateX #1 \n_rlnCoordinateY #2 \n_rlnAngleTiltPrior #3 \n'
	star_file = header
	for j in range(0, len(picks)):
		star_file = star_file + starify(picks[j][1]*4.0,picks[j][0]*4.0,90.0)
	
	star_file_name = 'starFiles/'+(big_micrograph_name[:-4].split('/')[-1]).split('_bin4')[0]
	with open(star_file_name+'.star', "w") as text_file:
		text_file.write(star_file)









