#!/home/alus_soft/matt_EMAN2/bin/python
####################################################################################################
from EMAN2 import *
import mrcfile
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
####################################################################################################
####################################################################################################
# Load the roughly aligned 3D volumes
rightFil_vol=EMData()
#rightFil_vol.read_image('rightFil_box448_matched_to2Dclass.mrc')
rightFil_vol.read_image('rightFil_currBest_it0_locSearch_20.mrc')

leftFil_vol=EMData()
#leftFil_vol.read_image('leftFil_box448_matched_to2Dclass.mrc')
leftFil_vol.read_image('leftFil_currBest_it0_locSearch_09.mrc')


#display 2D average to make sure it is correct
class_avg = EMData()
#class_avg.read_image('run_it040_classes_corrPxSize.mrcs', 0)
class_avg.read_image('refined_2Dclass_corrPxSize.mrc')
class_avg = class_avg.process('mask.ringmean', {'outer_radius':(350.0)/2.0})

####################################################################################################
# This function takes two 3D volumes as inputs. It adds them and projects them, then it masks the
# projection and compares to the 2D class average reference
def compute_loss_2Fils(fil_1, fil_2, class_avg):
	proj = (fil_1+fil_2).project('standard',Transform())
	proj_masked = proj.process('mask.ringmean', {'outer_radius':(350.0)/2.0})
	proj.process('normalize.toimage',{'to':proj_masked})
	#display([class_avg, proj_masked, proj])
	return(class_avg.cmp('ccc',proj_masked))

def save_projection(fil_1, fil_2, class_avg, file_name):
	proj = (fil_1+fil_2).project('standard',Transform())
	proj_masked = proj.process('mask.ringmean', {'outer_radius':(350.0)/2.0})
	proj_masked_np = EMNumPy.em2numpy(proj_masked)
	class_avg_np = EMNumPy.em2numpy(class_avg)
	#display([class_avg, proj_masked])
	with mrcfile.new(file_name,overwrite=True) as mrc:
		mrc.set_data(np.asarray([class_avg_np,proj_masked_np]))
	
	return

# This function takes a set of projections of 1 of the 2 filaments and adds the projection of the other
# filament's volume. This composite projection is then masked and compared to the 2D class average reference
def compute_losses(projected_fils, fil_2_vol, class_avg):
	cccs = []
	proj_fil2 = fil_2_vol.project('standard',Transform())
	print('Computing cross-correlational coefficient between projections and 2D class average')
	for i in tqdm(range(0, len(projected_fils))):
		proj = projected_fils[i] + proj_fil2
		proj_masked = proj.process('mask.ringmean', {'outer_radius':(350.0)/2.0})
		proj.process('normalize.toimage',{'to':proj_masked})
		cccs.append(class_avg.cmp('ccc',proj_masked))
	
	return cccs

# This function generates a set of rotated and translated projection images of a given filament with 
# defined step sizes
def sample_orientations(fil, ang_step, step_size, fil_ID):
	print('Generating set of rotated and translated projection images of the '+ fil_ID +' filament.')
	print('An angular step size of ' + str(ang_step) + ' degrees and a translational step size of ' + 
			str(step_size) +' pixels will be used')
	ts = []; fil_copies = []; projs = []; projs_trans = []; trans_holder = []
	print('\tCopying filament to array and generating 3D transform objects')
	for i in range(-1,2):
		for j in range(-1,2):
			for k in range(-1,2):
				ts.append(Transform({'type':'eman', 'alt':ang_step*i, 'az':ang_step*j, 'phi':ang_step*k}))
				fil_copies.append(fil.copy())
	
	print('\tApplying transformations to filament copies')
	for i in range(0,len(ts)):
		fil_copies[i].transform(ts[i])
	
	print('\tProjecting ' + str(len(ts)) + ' maps')
	for i in range(0, len(fil_copies)):
		projs.append(fil_copies[i].project('standard', Transform()))
	
	print('\tTranslating projections')
	for i in range(0,len(projs)):
		for j in range(-5,6):
			for k in range(-5,6):
				# Handle transformation parameters to know which params gave best result
				trans_params = {'tx':step_size*j, 'ty':step_size*k}
				full_trans_params = copy.deepcopy(ts[i])#.set_params(trans_params)
				full_trans_params.set_params(trans_params)
				trans_holder.append(full_trans_params)
				# Now, do xy translations to the projections and store
				temp_proj = projs[i].copy()
				temp_proj.transform(Transform(trans_params))
				projs_trans.append(temp_proj)
	
	return projs_trans, trans_holder

####################################################################################################
# Define some metaparameters, and initialize some values
curr_loss = 0; keep_going = True; ANG_STEP = 4; TRANS_STEP = 2.0; best_loss = 0 
currBest_leftFil_vol = leftFil_vol.copy(); currBest_rightFil_vol = rightFil_vol.copy()
curr_it = 0

print('The beginning loss is: ' + str(compute_loss_2Fils(currBest_leftFil_vol, currBest_rightFil_vol, class_avg)))
save_projection(currBest_leftFil_vol, currBest_rightFil_vol, class_avg, 'locSearch_refinedRef_initial.mrcs')
# Run the functions defined above iteratively
while(keep_going):
	left_fil_copies, trans_params = sample_orientations(currBest_leftFil_vol, ANG_STEP, TRANS_STEP, 'left')
	cccs = compute_losses(left_fil_copies, currBest_rightFil_vol, class_avg)
	updated_trans = trans_params[np.argmin(cccs)]
	print('The transformation to apply to the current best filament orientation is: ')
	print(updated_trans)
	currBest_leftFil_vol = currBest_leftFil_vol.copy()
	currBest_leftFil_vol.transform(updated_trans)
	
	curr_loss = compute_loss_2Fils(currBest_leftFil_vol, currBest_rightFil_vol, class_avg)
	print('The loss from the current iteration is: ' + str(curr_loss))
	if(curr_loss+0.0005 < best_loss):
		best_loss = curr_loss
	elif(TRANS_STEP>1.0):
		TRANS_STEP = TRANS_STEP/2.0
		print('Decreasing translational step size to: ' + str(TRANS_STEP))
	elif(ANG_STEP>0.5):
		ANG_STEP = ANG_STEP/2.0
		#TRANS_STEP = TRANS_STEP/2.0
		print('Decreasing angular step size to: ' + str(ANG_STEP))
	elif(TRANS_STEP>0.5):
		TRANS_STEP = TRANS_STEP/2.0
		print('Decreasing translational step size to: ' + str(TRANS_STEP))
	else:
		keep_going = False
		print('Converged for the left filament. Now updating the right filament')
	
	save_projection(currBest_leftFil_vol, currBest_rightFil_vol, class_avg, 'locSearch_refinedRef_leftFil_'+str(curr_it).zfill(2)+'.mrcs')
	currBest_leftFil_vol.write_image(  'leftFil_currBest_it1_locSearch_'+str(curr_it).zfill(2)+'.mrc')
	curr_it = curr_it+1

print('The best loss after fitting the first filament is: ' + str(compute_loss_2Fils(currBest_leftFil_vol, currBest_rightFil_vol, class_avg)))
#save_projection(currBest_leftFil_vol, currBest_rightFil_vol, class_avg, 'locSearch_leftFil_0.mrcs')

# Switch to the right filament
keep_going = True; ANG_STEP = 4; TRANS_STEP = 2.0;
while(keep_going):
	right_fil_copies, trans_params = sample_orientations(currBest_rightFil_vol, ANG_STEP, TRANS_STEP, 'right')
	cccs = compute_losses(right_fil_copies, currBest_leftFil_vol, class_avg)
	updated_trans = trans_params[np.argmin(cccs)]
	print('The transformation to apply to the current best filament orientation is: ')
	print(updated_trans)
	currBest_rightFil_vol = currBest_rightFil_vol.copy()
	currBest_rightFil_vol.transform(updated_trans)
	
	curr_loss = compute_loss_2Fils(currBest_rightFil_vol, currBest_leftFil_vol, class_avg)
	print('The loss from the current iteration is: ' + str(curr_loss))
	if(curr_loss+0.0005 < best_loss):
		best_loss = curr_loss
	elif(TRANS_STEP>1.0):
		TRANS_STEP = TRANS_STEP/2.0
		print('Decreasing translational step size to: ' + str(TRANS_STEP))
	elif(ANG_STEP>0.5):
		ANG_STEP = ANG_STEP/2.0
		#TRANS_STEP = TRANS_STEP/2.0
		print('Decreasing angular step size to: ' + str(ANG_STEP))
	elif(TRANS_STEP>0.5):
		TRANS_STEP = TRANS_STEP/2.0
		print('Decreasing translational step size to: ' + str(TRANS_STEP))
	else:
		keep_going = False
		print('Converged for the current filament')
	save_projection(currBest_leftFil_vol, currBest_rightFil_vol, class_avg, 'locSearch_refinedRef_rightFil_'+str(curr_it).zfill(2)+'.mrcs')
	currBest_rightFil_vol.write_image('rightFil_currBest_it1_locSearch_'+str(curr_it).zfill(2)+'.mrc')
	curr_it = curr_it+1

print('The best loss after fitting the second filament is: ' + str(compute_loss_2Fils(currBest_leftFil_vol, currBest_rightFil_vol, class_avg)))
#save_projection(currBest_leftFil_vol, currBest_rightFil_vol, class_avg, 'locSearch_rightFil_0.mrcs')














"""
left_fil_copies, trans_params = sample_orientations(leftFil_vol, ANG_STEP, TRANS_STEP, 'left')
cccs = compute_losses(left_fil_copies, rightFil_vol, class_avg)
updated_trans = trans_params[np.argmin(cccs)]
currBest_leftFil_vol = leftFil_vol.copy()
currBest_leftFil_vol.transform(updated_trans)

# Verify results are improving
compute_loss_2Fils(leftFil_vol, rightFil_vol, class_avg)
curr_loss = compute_loss_2Fils(currBest_leftFil_vol, rightFil_vol, class_avg)
print('The loss from the current iteration is: ' + str(curr_loss))
if(curr_loss+0.00001 < best_loss):
	best_loss = curr_loss
elif(ANG_STEP>0.5):
	ANG_STEP = ANG_STEP/2.0
	#TRANS_STEP = TRANS_STEP/2.0
	print('Decreasing angular step size to: ' + str(ANG_STEP))
else:
	keep_going = False
"""


