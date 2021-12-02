# Data processing scripts for plastin bundles
This repository is a collection of scripts used to process cryo-EM data of F-actin bundled by plastin. 

These scripts were developed by Matthew Reynolds.
A user-friendly graphical user interface is currently being developed using variations of these scripts and will be made available in the future.

## Associated Manuscripts:

Pre-print: 

Insert biorxiv here.


Publication: Will be updated upon peer-review and publication.


## Dependencies:
Several external python libraries were used in the scripts contained in this repository. If one would like to use or modify these scripts, it is suggested to create an anaconda environment with the following packages:
numpy, prody, sklearn, tqdm, matplotlib, mrcfile, scipy, skimage, keras, tensorflow, and EMAN2.

.yml files are provided in misc that describes the versions used to run these scripts. The path to one's own anaconda environment will need to be modified to run the scripts properly. Additionally, hard-coded paths to files will need to be modified in the source code.



## Description of contents:
**generate_synthetic_data:**
Contains the script used to generate synthetic projection images for network training. A 3D volume file is required to run the script; a link to this volume is provided in the google drive link in "Additional files".

**train_networks:**
Contains the scripts to train the denoising autoencoder network and the semantic segmentation network. The trained networks are provided in the google drive link in "Additional files".

**bundle_picking:**
Contains the script used to pick on micrographs using the trained networks. A synthetic projection used during training is provided for histogram normalization. Also, a script used to adjust particle picks from semantically-segmented micrographs is provided.

**projection_matching:**
Contains the scripts used to perform projection matching, in conjunction with EMAN2's coarse, global search.

**angle_measurements:**
Contains the scripts used to measure the splay and skew angles between bundled filaments.

**misc:**
Contains miscellaneous scripts for FRC measurements and duplicate removal. 


## Additional files:
Due to github's 25MB file limit size, the trained networks and 3D volume for generating synthetic data are hosted at this google drive link:
https://drive.google.com/drive/folders/160jktTYF-jqsUpWrS8wX_fWfiBngnTPg?usp=sharing






