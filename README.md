# Data processing scripts for plastin bundles
This repository is a collection of scripts used to process cryo-EM data of F-actin bundled by plastin. These scripts were developed by Matthew Reynolds.
A user-friendly graphical user interface is currently being developed using variations of these scripts and will be made available in the future.

## Associated Manuscripts:
Pre-print:
Insert biorxiv here.

Publication:
Will be updated upon peer-review and publication.


## Dependencies:
Several external python libraries were used in the scripts contained in this repository. If one would like to use or modify these scripts, it is suggested to create an anaconda environment with the following packages:
numpy
prody
sklearn
tqdm
matplotlib
mrcfile
scipy
skimage
keras
tensorflow
EMAN2
A .yml file is provided in misc that describes the versions used to run these scripts. The path to one's own anaconda environment will need to be modified to run the scripts properly.



## Description of contents:
generate_synthetic_data:
Contains the script used to generate synthetic projection images for network training.

train_networks:
Contains the scripts to train the denoising autoencoder network and the semantic segmentation network.

bundle_picking:
Contains the script used to pick on micrographs using the trained networks.

projection_matching:
Contains the scripts used to perform projection matching, in conjunction with EMAN2's coarse, global search.

angle_measurements:
Contains the scripts used to measure the splay and skew angles between bundled filaments.







