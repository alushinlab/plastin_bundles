#!/home/alus_soft/matt_EMAN2/bin/python
################################################################################
# import of python packages
print('Beginning to import packages...')
import numpy as np
import matplotlib.pyplot as plt
import keras
import mrcfile
import random
from tqdm import tqdm
from keras import layers
from keras.models import Model
import tensorflow as tf; import keras.backend as K
from scipy import interpolate; from scipy.ndimage import filters
#from helper_manifold_exploration import *
from keras.utils import to_categorical
from EMAN2 import *
print('Packages finished importing. Data will now be loaded')
################################################################################
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
################################################################################
# method to import synthetic data from files
def import_synth_data(noise_folder, noNoise_folder, box_length, NUM_IMGS_MIN, NUM_IMGS_MAX):
	noise_holder = []; noNoise_holder = []
	print('Loading files from ' + noise_folder)
	for i in tqdm(range(NUM_IMGS_MIN, NUM_IMGS_MAX)):
		file_name = 'actin_rotated%05d.mrc'%i
		noise_data = None; noNoise_data = None
		with mrcfile.open(noise_folder + file_name) as mrc:
			if(mrc.data.shape == (box_length,box_length)):
				noise_data = mrc.data
		with mrcfile.open(noNoise_folder + file_name) as mrc:
			if(mrc.data.shape == (box_length,box_length)):
				noNoise_data = mrc.data
				
		if(not np.isnan(noise_data).any() and not np.isnan(noNoise_data).any()): #doesn't have a nan
			noise_holder.append(noise_data.astype('float16'))
			noNoise_holder.append(noNoise_data.astype('float16'))
		
		else: # i.e. if mrc.data does have an nan, skip it and print a statement
			print('Training image number %d has at least one nan value. Skipping this image.'%i)
	
	return noise_holder, noNoise_holder

################################################################################
import keras.backend as K
def CCC(y_pred, y_true):
	x = y_true
	y = y_pred
	mx=K.mean(x)
	my=K.mean(y)
	xm, ym = x-mx, y-my
	r_num = K.sum(tf.multiply(xm,ym))
	r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
	r = r_num / r_den
	return -1*r

################################################################################
folder = './'
noise_folder = folder + 'tplastin_val_10k_3FilUnit_noise/'
noNoise_folder = folder + 'tplastin_val_10k_3FilUnit_noNoise/'

train, target = import_synth_data(noise_folder, noNoise_folder, 192, 0, 10000)
train = np.asarray(train, dtype='float16'); target = np.asarray(target,dtype='float16')

#add extra dimension at end because only one color channel
train = np.expand_dims(train, axis=-1)
target = np.expand_dims(target, axis=-1)

# Because the first 10% of the synth data was used as a validation set, I only 
# need to load in the first 10% of the data
#FRAC_VAL = train.shape[0]#int(train.shape[0] * 0.1)
#val_train = train[:FRAC_VAL]
#val_target = target[:FRAC_VAL]
#train = train[FRAC_VAL:]
#target = target[FRAC_VAL:]
print('All files loaded and parsed into training and validation sets.')
################################################################################
######### The data should be imported; now create the model ####################
################################################################################
# Import the encoding layers of the DAE model
model_path = './300000training_tplastin_CCC09856.h5'
autoencoder_three = keras.models.load_model(model_path, custom_objects={'CCC':CCC})

# Now run a prediction
preds = autoencoder_three.predict(train)[:,:,:,0]


fig, ax = plt.subplots(1,2)
ax[0].imshow(preds[5], cmap=plt.cm.gray)
ax[1].imshow(target[5,:,:,0].astype('float32'), cmap=plt.cm.gray)
plt.show()


with mrcfile.new('predicted_stack_10k_3Fils_2.mrcs') as mrc:
	mrc.set_data(preds.astype('float32'))


## Open now in /home/alus_soft/matt_EMAN2 anaconda environment
with mrcfile.open('predicted_stack_10k_3Fils_2.mrcs') as mrc:
	preds = mrc.data

target = target[:,:,:,0]

mask = EMData('./soft_mask_192_144_10.mrc')
FSCs = []
for i in tqdm(range(0, len(target))):
	predicted = EMNumPy.numpy2em(preds[i])*mask
	ground_truth = EMNumPy.numpy2em(target[i])* mask
	fsc=ground_truth.calc_fourier_shell_correlation(predicted)	# works in 2-D and 3-D, returns a single array with s,FSC,nvoxels
	third=len(fsc)/3
	saxis=fsc[0:third]			# spatial frequencies
	fscval=fsc[third:third*2]		# FSC/FRC values
	nvox=fsc[third*2:third*3]		# number of voxels used for each value
	FSCs.append([saxis, fscval])

FSCs = np.asarray(FSCs)
for i in tqdm(range(0,100)):#len(FSCs))):
	_=plt.plot(FSCs[i][0], FSCs[i][1], alpha=0.05, c='blue')

_=plt.plot(FSCs[i][0], np.average(FSCs[:,1], axis=0))
plt.ylim(-0.1,1)
plt.fill_between(FSCs[i][0], np.average(FSCs[:,1], axis=0) - np.std(FSCs[:,1], axis=0), np.average(FSCs[:,1], axis=0)+np.std(FSCs[:,1], axis=0), alpha=0.2)
plt.show()


np.savetxt('avgFRC_10k_eitherFilUnit_4units.txt', np.array((FSCs[i][0], np.average(FSCs[:,1], axis=0))))

# save text for lin to plot
col_1 = np.expand_dims(FSCs[0,0], 0)
col_2 = np.expand_dims((np.average(FSCs[:,1], axis=0) - np.std(FSCs[:,1], axis=0)),0)
col_3 = np.expand_dims((np.average(FSCs[:,1], axis=0)),0)
col_4 = np.expand_dims((np.average(FSCs[:,1], axis=0) + np.std(FSCs[:,1], axis=0)),0)
col_5 = FSCs[:100,1]

full_stack = np.vstack((col_1, col_2, col_3, col_4, col_5))

np.savetxt('xaxis_-std_avg_+std_100examples.txt', full_stack)


################################################################################
# Get FRC for one particular image
i = 5
mask = EMNumPy.em2numpy(EMData('../../data_storage_bent_actin128/validation/spiderstuff/softcircle_3.mrc'))[4:-4,4:-4]
mask = np.pad(mask, (4,4), 'constant', constant_values=0)
predicted = EMNumPy.numpy2em(np.multiply(preds[i], mask))
ground_truth = EMNumPy.numpy2em(np.multiply(val_target[i,:,:,0].astype('float32'), mask))
fsc=ground_truth.calc_fourier_shell_correlation(predicted)	# works in 2-D and 3-D, returns a single array with s,FSC,nvoxels
third=len(fsc)/3
saxis=fsc[0:third]			# spatial frequencies
fscval=fsc[third:third*2]		# FSC/FRC values
nvox=fsc[third*2:third*3]		# number of voxels used for each value
plt.plot(saxis, fscval)
plt.show()





