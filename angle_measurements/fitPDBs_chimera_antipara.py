##!/home/alus_soft/matt_EMAN2/bin/python
####################################################################################################
import os
import sys
from chimera import runCommand as rc # use 'rc' as shorthand for runCommand
####################################################################################################
# Load the particular *_boxed.mrc file
map_file_name = sys.argv[1]#'analyse_part011111_boxed.mrc'#


# Load starting PDBs; initial position same for all
fil1_file_name = './fil1_plastinDecoratedActin_antipara.pdb'
fil2_file_name = './fil2_plastinDecoratedActin_antipara.pdb'

fil1_new_fil_file_name = map_file_name[:-4] + '_fil1.pdb'
fil2_new_fil_file_name = map_file_name[:-4] + '_fil2.pdb'

# Perform chimera operations:
# Open files
rc('open ' + map_file_name)
rc('open ' + fil1_file_name)
rc('open ' + fil2_file_name)


# Adjust voxel size and do fitting
rc('volume #0 voxelSize 5.49333333')
rc('volume #0 level 0.003')
rc('fitmap #1 #0')
rc('fitmap #2 #0')

# Save PDBs
rc('write format pdb relative #0 #1 ' + fil1_new_fil_file_name)
rc('write format pdb relative #0 #2 ' + fil2_new_fil_file_name)

# Exit script
rc("stop now")


