#!/home/greg/.conda/envs/matt_TF/bin/python
####################################################################################################
from prody import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import glob
from tqdm import tqdm
import sys
####################################################################################################
chunk_num = str(sys.argv[1])
####################################################################################################
# Measure distance between 482G on each filament, effective bridge length
# Location of 482G on plastins
def compute_bridge_length(fil1, fil2):
	fil1_482G_coord = fil1.select('chain K').getCoords()[463]
	fil2_482G_coord = fil2.select('chain L').getCoords()[463]
	bridge_length = np.linalg.norm(fil1_482G_coord - fil2_482G_coord)
	return bridge_length

# Use PCA to determine central axis of reference filament, use only actin alpha carbons
def get_central_axis(fil):
	pca = PCA(n_components=3)
	pca.fit(fil.select('resnum < 368').getCoords())
	fil_COM = np.average(fil.select('resnum < 368').getCoords(), axis=0)
	fil_central_axis = pca.components_[0]
	return fil_central_axis, fil_COM

# Project fil1_central axis and fil2_central axis to plane normal to normal_vect to get splay views
# Reference for projection of vector to plane
# https://www.maplesoft.com/support/help/maple/view.aspx?path=MathApps%2FProjectionOfVectorOntoPlane
# proj_plane(u) = u  - proj_n(u) = u - (dot(u,n)/||n||^2)*n
def proj_filament_to_plane(u,n):
	projected_fil = u - (np.dot(u, n)/(np.linalg.norm(n)**2))*n
	return projected_fil

# Return the splay or skew angles between two filaments
def get_splay_or_skew_angle(fil1_central_axis, fil2_central_axis, n):
	proj_fil1_splay = proj_filament_to_plane(fil1_central_axis, n)
	proj_fil2_splay = proj_filament_to_plane(fil2_central_axis, n)
	splay_angle = np.degrees(np.arccos(np.dot(proj_fil1_splay, proj_fil2_splay)/(np.linalg.norm(proj_fil1_splay)*np.linalg.norm(proj_fil2_splay))))
	cross_projs = np.cross(proj_fil1_splay,proj_fil2_splay)
	if(np.dot(cross_projs, n)>0):
		return splay_angle
	else:
		return -1.0*splay_angle

# Return the minimum distance between a reference filament's center of mass and the mobile filament's
# central axis https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
# p = pt off line we want shortest distance from; n = vector defining line direction; a = pt on line
def get_interfil_distance(p, n, a):
	return (np.linalg.norm(np.cross((a-p),n)) / np.linalg.norm(n))

####################################################################################################
def measure_angles_and_dist(fil1_name, fil2_name):
	# Load the pdb files
	fil1 = parsePDB(fil1_name, subset='ca')
	fil2 = parsePDB(fil2_name, subset='ca')
	
	# Use PCA to determine central axis of both filaments, use only actin alpha carbons
	fil1_central_axis, fil1_COM = get_central_axis(fil1)
	fil2_central_axis, fil2_COM = get_central_axis(fil2)
	
	# Define projection angle: COM of the actins to V139 on chain L of fil1 349
	# subtract a fraction of z to make normal vector perpendicular
	fil1_139V_coord = fil2.select('chain L').getCoords()[343]-[0.0,0.0,0.7837] 
	
	# Define vectors orthogonal to the reference filament's central axis and each other
	normal_vect = fil2_COM - fil1_139V_coord
	new_normal = np.cross(fil2_central_axis, normal_vect)
	
	# Perform measurements
	bridge_length = compute_bridge_length(fil1, fil2)
	splay_angle = get_splay_or_skew_angle(fil1_central_axis, fil2_central_axis, normal_vect)
	skew_angle = get_splay_or_skew_angle(fil1_central_axis, fil2_central_axis, new_normal)
	interfil_dist = get_interfil_distance(fil1_COM, fil2_central_axis, fil2_COM)
	
	return bridge_length, splay_angle, skew_angle, interfil_dist

####################################################################################################

fil1_file_names = sorted(glob.glob('../parallel/chunk_%s/*fil1.pdb'%(chunk_num)))
fil2_file_names = sorted(glob.glob('../parallel/chunk_%s/*fil2.pdb'%(chunk_num)))
results = []
for i in tqdm(range(0,len(fil1_file_names))):
	results.append(measure_angles_and_dist(fil1_file_names[i], fil2_file_names[i]))

results = np.asarray(results)

#np.savetxt(results,'dist_splay_skew_chunk_01.txt')
np.savetxt('./results_txt_corrChain_final/dist_splay_skew_chunk_%s.txt'%(chunk_num),results)

'''
_=plt.hist(results[:,0],bins=50)
plt.show()


_=plt.hist(results[:,1],bins=50)
plt.show()

_=plt.hist(results[:,2]-180,bins=50)
plt.show()

_=plt.hist(results[:,3],bins=30)
plt.show()
'''









