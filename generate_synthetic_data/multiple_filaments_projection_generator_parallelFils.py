#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# imports
import argparse; import sys
parser = argparse.ArgumentParser('Generate noisy and noiseless 2D projections of randomly oriented and translated MRC files.')
parser.add_argument('--input_mrc_dir', type=str, help='input directory containing MRC files to be rotated, translated, CTF-convolved, and projected')
parser.add_argument('--output_noise_dir', type=str, help='output directory to store noisy 2D projections')
parser.add_argument('--output_noiseless_dir', type=str, help='output directory to store noiseless 2D projections')
parser.add_argument('--output_semMap_dir', type=str, help='output directory to store semantic segmentation targets')
parser.add_argument('--numProjs', type=int, help='total number of projections to make')
parser.add_argument('--nProcs', type=int, help='total number of parallel threads to launch')
args = parser.parse_args()
print('')
if(args.input_mrc_dir == None or args.output_noise_dir == None or args.output_noiseless_dir == None):
	print('Please enter an input_mrc_dir, AND an output_noise_dir, AND an output_noiseless_dir')
	sys.exit('The preferred input style may be found with ./projection_generator.py -h')

if(args.numProjs == None):
	sys.exit('Please enter the number of projection images you would like with the --numProjs flag')

if(args.nProcs == None):
	print('No process number specified, using one thread')
	nProcs = 1
else:
	nProcs = args.nProcs

if(args.numProjs % nProcs != 0):
	print('The numProjs that you specified was not a multiple of nProcs.')
	print('Instead of %d 2D projections, this program will generate %d 2D projections'%(args.numProjs, args.numProjs/nProcs*nProcs))

folder = args.input_mrc_dir
noNoise_outputDir = args.output_noiseless_dir
noise_outputDir = args.output_noise_dir
semMap_outputDir = args.output_semMap_dir
TOTAL_NUM_TO_MAKE = args.numProjs

if(folder[-1] != '/'): folder = folder + '/'
if(noNoise_outputDir[-1] != '/'): noNoise_outputDir = noNoise_outputDir + '/'
if(noise_outputDir[-1] != '/'): noise_outputDir = noise_outputDir + '/'
if(semMap_outputDir[-1] != '/'): semMap_outputDir = semMap_outputDir + '/'
print('The program will now generate %d 2D projections'%(args.numProjs/args.nProcs*args.nProcs))
################################################################################
# import of python packages
import numpy as np
from EMAN2 import *; from sparx import *; import mrcfile
import json; import glob
from multiprocessing import Pool
import os; from tqdm import tqdm
from skimage.morphology import erosion,dilation; from skimage.morphology import disk
################################################################################
################################################################################
# import data
actin_orig = []
file_names = sorted(os.listdir(folder))
for file_name in file_names:
	if(file_name[-4:] == '.mrc' and file_name[:10] == 'bent_actin'):
		actin_orig.append(EMData(folder+file_name))

################################################################################
# image sizes
box_len = 384; BL = 96; cropBox = BL*2
################################################################################
def launch_parallel_process(thread_idx):
	index=num_per_proc*thread_idx
	for i in tqdm(range(0,num_per_proc)):
		local_random_state = np.random.RandomState(None)
		# First: randomly pick one of the actin mrc files that were loaded into actin_orig
		r0 = local_random_state.randint(0,len(actin_orig))
		num_filaments = local_random_state.randint(0,4)
		if(num_filaments == 0):
			target = np.concatenate((np.ones((1,cropBox,cropBox)), np.zeros((2,cropBox,cropBox))), axis=0)
			with mrcfile.new(semMap_outputDir + 'actin_rotated%05d.mrcs'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target.astype('float32'))
			
			r7 = local_random_state.uniform(1.0, 4.0) #defocus 
			r8 = max(local_random_state.normal(0.050, 0.010),0) # noise amplitude
			target_eman = EMNumPy.numpy2em(np.zeros((cropBox,cropBox)))
			target_eman.process_inplace('math.simulatectf',{'ampcont':10.0,'apix':4.12,'bfactor':0.0,'cs':2.7,'defocus':r7,'noiseamp':r8,'purectf':False,'voltage':300.0})
			target_noise = EMNumPy.em2numpy(target_eman)
			with mrcfile.new(noise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target_noise.astype('float32'))
			with mrcfile.new(noNoise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target[1].astype('float32'))		

		else:
			target = np.zeros((5,cropBox,cropBox)); target_lp50 = np.zeros((5,cropBox,cropBox))
			bundle_idxs = []
			for j in range(1, num_filaments+1):
				r0_name = file_names[r0]
				rotated_actin = actin_orig[r0].copy()
				# handle bundles
				r9 = local_random_state.uniform()
				fil_is_bundle = r9 < 0.65
				if(fil_is_bundle): # i.e. if this filament-unit is a bundle
					r10 = int(local_random_state.normal(loc=90, scale=1.5) + int(local_random_state.randint(0,2)*180))-90 #
					r11 = int(local_random_state.random_sample()*360) #
					r12 = local_random_state.normal(loc=36.8, scale=3.2) # bundle distance, measured by Alfred
					r13 = local_random_state.uniform(-44,44) # should cover full crossover
					rotated_actin2 = actin_orig[r0].copy()
					t2 = Transform()
					t2.set_params({'type':'eman', 'az':0, 'alt':r10, 'phi':r11, 'tx':0, 'ty':r12, 'tz':r13}) #tz is shift up and down filament axis # y and x control distance, alt is (anti-)parallel
					rotated_actin2.transform(t2)
					rotated_actin.add(rotated_actin2)
					bundle_idxs.append(j)
				 
				# Rotation angles: azimuth, alt, phi, then Translations: tx, ty,tz
				r1, r2, r3 = int(local_random_state.random_sample()*360),int(local_random_state.normal(loc=90, scale=2.5)),int(local_random_state.random_sample()*360)
				#r4, r5, r6 = local_random_state.normal(0, 25), local_random_state.normal(0, 25), local_random_state.normal(0, 25)
				r4, r5, r6 = local_random_state.uniform(-60, 60), local_random_state.uniform(-60, 60), local_random_state.uniform(-60, 60)
				t = Transform()
				t.set_params({'type':'eman','az':r1, 'alt':r2, 'phi':r3, 'tx':r4, 'ty':r5, 'tz':r6})
				rotated_actin.transform(t) # apply rotation and translation
				proj_eman = rotated_actin.project('standard',Transform()) # project
				proj_eman_lp50 = proj_eman.process('filter.lowpass.gauss', {'apix':4.12, 'cutoff_freq':1.0/40.0})
				proj_np_lp50 = EMNumPy.em2numpy(proj_eman_lp50)
				proj_np_lp50 = dilation(proj_np_lp50, selem=disk(8))
				proj_np_lp50 = erosion(proj_np_lp50, selem=disk(16))
				proj_np = EMNumPy.em2numpy(proj_eman)
				center = box_len/2
				# Save the target image
				target_filament = proj_np[center-BL:center+BL, center-BL:center+BL]
				#target_filament = (target_filament - np.mean(target_filament)) / np.std(target_filament) # normalize
				target[j] = target_filament
				
				target_filament_lp50 = proj_np_lp50[center-BL:center+BL, center-BL:center+BL]
				target_filament_lp50 = (target_filament_lp50 - np.mean(target_filament_lp50)) / np.std(target_filament_lp50) # normalize
				target_lp50[j] = target_filament_lp50
			
			
			# Generate noisy image
			target_proj = np.sum(target, axis=0)
			target_proj = (target_proj - np.mean(target_proj)) / np.std(target_proj)
			r7 = local_random_state.uniform(1.0, 4.0) #defocus
			r8 = max(local_random_state.normal(0.027, 0.005),0) # noise amplitude 0.050, 0.010 is default
			target_eman = EMNumPy.numpy2em(target_proj)
			target_eman.process_inplace('math.simulatectf',{'ampcont':10.0,'apix':4.12,'bfactor':0.0,'cs':2.7,'defocus':r7,'noiseamp':r8,'purectf':False,'voltage':300.0})
			target_noise = EMNumPy.em2numpy(target_eman)
			target_noise = (target_noise - np.mean(target_noise)) / np.std(target_noise) # normalize
			with mrcfile.new(noise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target_noise.astype('float32'))
			with mrcfile.new(noNoise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target_proj.astype('float32'))
			
			# make masks from each proj
			for j in range(1, num_filaments+1):
				target_lp50[j] = (target_lp50[j]>0)
			temp = np.sum(target_lp50, axis=0)
			
			semMap = np.array(np.zeros((3,cropBox,cropBox)))
			semMap[0] = (temp==0)>0
			#semMap[1] = -1*semMap[0] + 1
			semMap[2] = np.sum(target_lp50[bundle_idxs], axis=0) > 0
			semMap[1] = -1*((semMap[0] + semMap[2]) - 1)
			with mrcfile.new(semMap_outputDir + 'actin_rotated%05d.mrcs'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(semMap.astype('float32'))


################################################################################
# run in parallel
num_per_proc = TOTAL_NUM_TO_MAKE / nProcs
if __name__ == '__main__':
	p=Pool(nProcs)
	p.map(launch_parallel_process, range(0, nProcs))

################################################################################
# Now all files are written, combine all json files into one master json file
read_files = glob.glob(noise_outputDir+'params_*.json')
output_list = []
for f in read_files:
	for line in open(f, 'r'):
		output_list.append(json.loads(line))

#sort the json dictionaries based on iteration number
output_list = sorted(output_list, key=lambda i: i['iteration'])
for line in output_list:
	with open(noise_outputDir+'master_params.json', 'a') as fp:
		data_to_write = json.dumps(line)
		fp.write(data_to_write + '\n')


