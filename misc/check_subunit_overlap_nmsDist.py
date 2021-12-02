import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
################################################################################
file_name = 'parallel_particles.star'
micrographNum = np.loadtxt(file_name, dtype=str, skiprows=48, usecols=(1))
middle_checkXY = np.loadtxt(file_name, skiprows=48, usecols=(2,3,20,21))

micro_num = np.zeros((len(micrographNum)), dtype=int)
for i in range(0, len(micrographNum)):
	micro_num[i] = int(micrographNum[i][:-4].split('_')[1])

center_coord = middle_checkXY[:,:2] - middle_checkXY[:,2:]

################################################################################
################################################################################
# new way
per_micro_idxs = {}
for i in tqdm(set(micro_num)):
	micro_set = np.argwhere(micro_num==i)[:,0]
	per_micro_idxs[i] = micro_set

indices_to_keep = []
for key, value in tqdm(per_micro_idxs.iteritems()):
	value_orig = value.copy()
	project_dist_matrix=1
	while(np.sum(project_dist_matrix) !=0):
		sq_len = len(value)
		coord1 = np.repeat(np.expand_dims(center_coord[value], axis=0), sq_len, axis=0).reshape(sq_len,sq_len,2)
		coord2 = np.repeat(np.expand_dims(center_coord[value], axis=-1), sq_len, axis=0).reshape(sq_len,sq_len,2)
		dist_matrix = np.linalg.norm(coord1 - coord2, axis=-1)<(400)
		dist_matrix_binarized = np.multiply(-1*np.eye(len(dist_matrix))+1,dist_matrix)
		project_dist_matrix = np.sum(dist_matrix_binarized,axis=0)
		val_to_pop = np.argmax(project_dist_matrix)
		value = np.delete(value, val_to_pop)
	
	indices_to_keep = indices_to_keep + list(value)


indices_to_keep = np.asarray(indices_to_keep).flatten()
################################################################################
################################################################################
# Prepare star file
star_header = np.loadtxt(file_name, dtype=str, delimiter='$')[:46]
star_body = np.loadtxt(file_name, dtype=str, delimiter='?', skiprows=48)[:-1]
star_body_new = star_body[indices_to_keep]
# Random subset 1 star file
new_star_file = list(star_header) + list(star_body_new) + list([' '])
cntr=-1
with open('400AnoOverlap_nmsDist_parallel_41k.star', "w") as f:
	for line in new_star_file:
		if(cntr>=1 and cntr<=26):
			f.write(line+ '#%d'%cntr +'\n')
		else:
			f.write(line+'\n')
		cntr=cntr+1





