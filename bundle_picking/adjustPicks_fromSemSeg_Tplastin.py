#!/home/greg/.conda/envs/matt_TF/bin/python
################################################################################
print('Loading python packages...')
################################################################################
# import of python packages
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import random
from tqdm import tqdm
from scipy import interpolate; from scipy.ndimage import filters
from skimage.morphology import skeletonize_3d; import scipy
import glob
import os
from scipy.ndimage.measurements import center_of_mass
import sys
################################################################################
GPU_IDX = int(sys.argv[1])
print('Using CPU index #' + str(GPU_IDX))
################################################################################
################# Functions for opterating on whole micrographs ################
def implement_NMS(semMap):
	################################################################################
	# Four parameters
	box_size = 96
	half_box_size = int(box_size / 2)
	occ_high = 0.65
	occ_low = 0.20 # 0.15
	radius = 26
	inner_box = 12
	inner_box_half = int(inner_box / 2)
	occ_smallBox = 0.9
	com_allowed_dist = 4
	################################################################################
	picks = np.argwhere(semMap==1)
	# picks_noborder means delete those points too close to the edge of the image
	picks_noborder = [] # make more efficient with np.where
	for x in range(0,picks.shape[0]):
		if picks[x][0] >= half_box_size and picks[x][1] >= half_box_size and picks[x][0] < semMap.shape[0]-half_box_size and picks[x][1] < semMap.shape[1]-half_box_size:
			picks_noborder.append(picks[x])
	
	picks_noborder = np.asarray(picks_noborder)
	
	# count how many pixels in a rastered window of sizex [box_size x box_size] are white
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
		if(np.abs(com[0]-half_box_size) < com_allowed_dist or np.abs(com[1]-half_box_size) < com_allowed_dist):
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
################################################################################
# Load real micrographs
file_names = sorted(glob.glob('../../bundled_Tplastin_NoCa/particle_picking/pngs_semSeg/*_bin4.mrc'))
img_num = len(file_names)#len(sorted(glob.glob('../Micrographs_bin4/*_bin4.mrc')))
img_start = (img_num / 30)*GPU_IDX
img_end = (img_num / 30)*(GPU_IDX+1)
file_names = file_names[img_start:img_end]
file_names_2 = sorted(glob.glob('../Micrographs_bin4/*_bin4.mrc'))
file_names_2 = file_names_2[img_start:img_end]

print(img_start, img_end, len(file_names))
print('Using semantic segmentation map from ' + file_names[0] + ' to ' + file_names[-1])
print('Updating particle coordinates from ' + file_names_2[0] + ' to ' + file_names_2[-1])

for i in tqdm(range(0, len(file_names))):
	big_micrograph_name = file_names[i]
	with mrcfile.open(big_micrograph_name) as mrc:
		binarized_stitch_back = mrc.data
	
	picks = implement_NMS(binarized_stitch_back)
	
	big_micrograph_name = file_names_2[i]
	with mrcfile.open(big_micrograph_name) as mrc:
		binarized_stitch_back = mrc.data
	
	_=plt.imshow(binarized_stitch_back, origin='lower', cmap=plt.cm.gray)
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









