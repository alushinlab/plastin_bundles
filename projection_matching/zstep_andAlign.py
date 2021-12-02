#!/home/alus_soft/matt_EMAN2/bin/python
####################################################################################################
from EMAN2 import *
import mrcfile
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
####################################################################################################
def stepZ_projAlign(right_fil, left_fil, t_p, z, class_avg_EM):
	# step in z
	e = right_fil.copy()
	tz = Transform()
	tz.set_params({'type':'eman','tz':z})
	e.process_inplace("xform",{"transform":tz})
	
	antiparallel_combo = (left_fil + e)
	#t_p = Transform({'type':'eman','az':180.910,'alt':70})
	combo_proj = antiparallel_combo.project('standard',t_p) 
	
	# Align translated 2D class average to the original from the right filament to the untranslated 2D 
	# class average  and save the transformation operation as rFil_T
	combo_proj = combo_proj.align('rotate_translate', class_avg_EM,{'maxshift':50})
	combo_T = combo_proj.get_attr('xform.align2d')
	
	#combo_T_refined = combo_proj.align('refine', class_avg_EM, {'maxshift':50}).get_attr('xform.align2d')
	combo_proj = combo_proj.align('refine', class_avg_EM, {'maxshift':50})
	combo_T_refined = combo_proj.get_attr('xform.align2d')
	
	# Display projected image, original class average, and translated class average
	combo_proj_mask = combo_proj.process('mask.ringmean', {'outer_radius':(390.0)/2.0})
	combo_proj_mask = combo_proj_mask.process('normalize.toimage',{'to':class_avg_EM})
	#display((combo_proj, class_avg_EM, combo_proj_mask))
	masked_CCC = (class_avg_EM.cmp('ccc',combo_proj_mask))
	
	return [z, masked_CCC, combo_T, combo_T_refined, combo_proj_mask]

####################################################################################################
# import 2D class average
with mrcfile.open('run_classes_corrPxSize.mrcs') as mrc:
	class_avg = mrc.data

#display 2D average to make sure it is correct
plt.imshow(class_avg, cmap=plt.cm.gray)
plt.show()

class_avg_EM = EMNumPy.numpy2em(class_avg)
class_avg_EM = class_avg_EM.process('normalize')

####################################################################################################
# Load the aligned filaments
left_fil=EMData()
left_fil.read_image('leftFil_currBest_it1_locSearch_10.mrc')
right_fil=EMData()
# right_fil.read_image('rightFil_currBest_it1_locSearch_19.mrc') #zeroeth iteration
right_fil.read_image('rightFil_currBest_it1_locSearch_15_tzCoarse.mrc') # first iteration


#t_p = Transform({'type':'eman','az':189.284,'alt':50}) #zeroeth iteration
t_p = Transform({'type':'eman','az':189.220,'alt':55}) # first iteration
zStep_results = []
for i in tqdm(range(-50, 51)):
	zStep_results.append(stepZ_projAlign(left_fil, right_fil, t_p, i, class_avg_EM))

CCCs = []; zStep_imgs = []
for i in range(0, len(zStep_results)):
	CCCs.append(zStep_results[i][1])
	zStep_imgs.append(EMNumPy.em2numpy(zStep_results[i][4]))

with mrcfile.new('zStepProjs_it1.mrcs',overwrite=True) as mrc:
	mrc.set_data(np.asarray(zStep_imgs))


# Now apply transformations to 3D maps
right_fil_map = right_fil.copy()
tz = Transform()
tz.set_params({'type':'eman','tz':zStep_results[np.argmin(CCCs)][0]})
right_fil_map.transform(tz)

right_fil_map.write_image('rightFil_currBest_it1_locSearch_15_tzCoarseit1.mrc')




# generate text file with CCCs
cccs_holder = []
for i in range(0, len(zStep_results)):
	cccs_holder.append(zStep_results[i][:2])

cccs_holder = np.asarray(cccs_holder)









