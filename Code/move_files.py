from pathlib import Path
import os

"""
- consolidate data into one directory
- https://www.nitrc.org/plugins/mwiki/index.php/neurobureau:AthenaPipeline
- Data to get: 
    - Functional
        - snwmrda{subject#}_session_{session#}_rest_{scan#}.nii.gz: Preprocessed resting state fMRI data, written into MNI space at 4 mm x 4 mm x 4 mm voxel resolution, nuisance variance removed [1,2], 
            - and blurred with a 6-mm FWHM Gaussian filter
        - sfnwmrda{subject#}_session_{session#}_rest_{scan#}.nii.gz: Preprocessed resting state fMRI data, written into MNI space at 4 mm x 4 mm x 4 mm voxel resolution, nuisance variance removed [1,2], 
            - filtered using a bandpass filter (0.009 Hz < f < 0.08 Hz) [2,3,4] and blurred with a 6-mm FWHM Gaussian filter
"""

                   
base_dir = "../data"
dataset_dir = "../data/model_data"

for filepath in Path(base_dir).glob('**/s*wmrda*session_*_rest_*.nii.gz'):
    str_split = str(filepath).split('/')
    origin = str_split[2]
    filename = str_split[-1]
    new_filename = origin + "_" + filename
    new_filepath = os.path.join(dataset_dir, new_filename)
    os.rename(filepath, new_filepath)