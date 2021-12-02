#!/home/alus_soft/matt_EMAN2/bin/python
####################################################################################################
from EMAN2 import *
import mrcfile
import matplotlib.pyplot as plt
import numpy as np
####################################################################################################
# import 2D class average
with mrcfile.open('run_it040_classes_corrPxSize.mrcs') as mrc:
	class_avg = mrc.data

#display 2D average to make sure it is correct
plt.imshow(class_avg, cmap=plt.cm.gray)
plt.show()

class_avg_EM = EMNumPy.numpy2em(class_avg)

####################################################################################################
# Load the 2D class average, shifted
right_fil_2Davg=EMData()
right_fil_2Davg.read_image('firstFil_10degAngSamp.hdf', 0)

# compute CCC before alignment, to gauge if it changes after alignment
print(class_avg_EM.cmp('ccc',right_fil_2Davg))

# Align translated 2D class average to the original from the right filament to the untranslated 2D 
# class average  and save the transformation operation as rFil_T
right_fil_2Davg = right_fil_2Davg.align('rotate_translate', class_avg_EM,{'maxshift':75})
rFil_T = right_fil_2Davg.get_attr('xform.align2d')

rFil_T_refined = right_fil_2Davg.align('refine', class_avg_EM, {'maxshift':10}).get_attr('xform.align2d')

# compute CCC after alignment to check that it changed
print(class_avg_EM.cmp('ccc',right_fil_2Davg))

# Load the projected and shifted image of the right filament
right_fil=EMData()
right_fil.read_image('firstFil_10degAngSamp.hdf', 1)

right_fil.get_attr('xform.projection')

# Apply translation determined from the alignment to the projection of the right filament
right_fil.transform(rFil_T)
right_fil.transform(rFil_T_refined)

# Display projected image, original class average, and translated class average
right_fil_mask = right_fil.process('mask.ringmean', {'outer_radius':(390.0)/2.0})
display((right_fil, class_avg_EM.process('normalize'), right_fil_2Davg, right_fil_mask))



right_fil_mask_norm = right_fil_mask.process('normalize.toimage', {'to':class_avg_EM.process('normalize')})


signal_subtracted = class_avg_EM.process('normalize') - right_fil_mask_norm
signal_subtracted.write_image('subtracted_first_fil_10deg.mrc')

display((right_fil, class_avg_EM.process('normalize'), right_fil_2Davg, right_fil_mask, right_fil_mask_norm,signal_subtracted))

####################################################################################################
# Load the 2D class average, shifted
left_fil_2Davg=EMData()
left_fil_2Davg.read_image('secondFil_10degAngSamp.hdf', 0)

signal_subtracted.read_image('subtracted_first_fil_30deg.mrc')

# compute CCC before alignment, to gauge if it changes after alignment
print(class_avg_EM.cmp('ccc',left_fil_2Davg))

# Align translated 2D class average to the original from the left filament to the untranslated 2D 
# class average and save the transformation operation as lFil_T
left_fil_2Davg = left_fil_2Davg.align('rotate_translate', signal_subtracted)
lFil_T = left_fil_2Davg.get_attr('xform.align2d')

lFil_T_refined = left_fil_2Davg.align('refine', signal_subtracted, {'maxshift':10}).get_attr('xform.align2d')

# compute CCC after alignment to check that it changed
print(class_avg_EM.cmp('ccc',left_fil_2Davg))

# Load the projected and shifted image of the left filament
left_fil=EMData()
left_fil.read_image('secondFil_10degAngSamp.hdf', 1)
left_fil.get_attr('xform.projection')

# Apply translation determined from the alignment to the projection of the left filament
left_fil.transform(lFil_T)
left_fil.transform(lFil_T_refined)

# Display projected image, original class average, and translated class average
display((left_fil, class_avg_EM.process('normalize'), left_fil_2Davg))

####################################################################################################
display((right_fil, class_avg_EM, right_fil_2Davg, left_fil, class_avg_EM, left_fil_2Davg))

composite_projection = (right_fil + left_fil).process('mask.ringmean', {'outer_radius':(390.0)/2.0}).process('normalize.toimage',{'to':class_avg_EM})

right_fil_2Davg_np = EMNumPy.em2numpy(right_fil)
left_fil_2Davg_np  = EMNumPy.em2numpy(left_fil)
composite_projection_np  = EMNumPy.em2numpy(composite_projection)
fig, ax = plt.subplots(1,2)
#ax[0].imshow(right_fil_2Davg_np + left_fil_2Davg_np, cmap=plt.cm.gray)
ax[0].imshow(composite_projection_np, cmap=plt.cm.gray)
ax[1].imshow(class_avg, cmap=plt.cm.gray)
plt.show()

# import 2D class average
with mrcfile.new('stackedProj_and2dAvg.mrcs',overwrite=True) as mrc:
	mrc.set_data(np.asarray([composite_projection_np, class_avg]))


# Now apply transformations to 3D maps
left_fil_map=EMData()
left_fil_map.read_image('postprocess_lp15_box448.mrc')

right_fil_map=EMData()
right_fil_map.read_image('postprocess_lp15_box448.mrc')

display((left_fil_map, right_fil_map))

left_fil_map.transform(left_fil.get_attr('xform.projection'))
left_fil_map.transform(lFil_T)
left_fil_map.transform(lFil_T_refined)

right_fil_map.transform(right_fil.get_attr('xform.projection'))
right_fil_map.transform(rFil_T)
right_fil_map.transform(rFil_T_refined)

left_fil_map.write_image('leftFil_box448_matched_to2Dclass.mrc')
right_fil_map.write_image('rightFil_box448_matched_to2Dclass.mrc')




