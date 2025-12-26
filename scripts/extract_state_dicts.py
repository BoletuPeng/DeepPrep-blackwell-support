#!/usr/bin/env python3
"""
extract_state_dicts.py

This script extracts state dictionaries from DeepPrep SUGAR models.
Run this script using the ORIGINAL DeepPrep container (pbfslab/deepprep:25.1.0)
to extract model parameters in a format that can be loaded by newer torch_geometric versions.

Usage:
    docker run --rm -v %cd%:/scripts -v %cd%\converted_models:/output \
        --entrypoint python pbfslab/deepprep:25.1.0 /scripts/extract_state_dicts.py
"""

import sys
sys.path.insert(0, '/opt/DeepPrep/deepprep/SUGAR')
import torch
import os

models = [
    '/opt/model/SUGAR/model_files/fsaverage6/lh_NoRigid_904_fsaverage6.model',
    '/opt/model/SUGAR/model_files/fsaverage6/rh_NoRigid_904_fsaverage6.model',
    '/opt/model/SUGAR/model_files/fsaverage6/rh_Rigid_904_fsaverage6.model',
    '/opt/model/SUGAR/model_files/fsaverage6/lh_Rigid_904_fsaverage6.model',
    '/opt/model/SUGAR/model_files/fsaverage4/lh_NoRigid_904_fsaverage4.model',
    '/opt/model/SUGAR/model_files/fsaverage4/rh_NoRigid_904_fsaverage4.model',
    '/opt/model/SUGAR/model_files/fsaverage3/lh_NoRigid_904_fsaverage3.model',
    '/opt/model/SUGAR/model_files/fsaverage3/rh_NoRigid_904_fsaverage3.model',
    '/opt/model/SUGAR/model_files/fsaverage5/lh_NoRigid_904_fsaverage5.model',
    '/opt/model/SUGAR/model_files/fsaverage5/rh_NoRigid_904_fsaverage5.model',
]

os.makedirs('/output/state_dicts', exist_ok=True)

for model_path in models:
    print(f'Processing {model_path}')
    m = torch.load(model_path, map_location='cpu')
    inner = m['model']
    sd = inner.state_dict()
    rigid = inner.rigid
    ico_level = [k for k in ['fsaverage3','fsaverage4','fsaverage5','fsaverage6'] if k in model_path][0]
    basename = os.path.basename(model_path).replace('.model', '.pt')
    out_path = f'/output/state_dicts/{basename}'
    torch.save({'state_dict': sd, 'rigid': rigid, 'ico_level': ico_level}, out_path)
    print(f'Saved to {out_path}')

print('Done!')
