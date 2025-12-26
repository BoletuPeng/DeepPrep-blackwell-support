#!/usr/bin/env python3
"""
rebuild_models.py

This script rebuilds DeepPrep SUGAR models using the NEW container environment.
Run this script using the upgraded container (with torch_geometric 2.7.0)
to create model files compatible with the new environment.

Usage:
    docker run --rm --gpus all -v %cd%:/scripts -v %cd%\converted_models:/input \
        -v %cd%\new_models:/output \
        --entrypoint /opt/conda/envs/deepprep/bin/python \
        deepprep:rtx5090-step4 /scripts/rebuild_models.py
"""

import sys
sys.path.insert(0, '/opt/DeepPrep/deepprep/SUGAR')
import torch
import os
from gatunet_model import GatUNet

input_dir = '/input/state_dicts'
output_dir = '/output'

for filename in os.listdir(input_dir):
    if not filename.endswith('.pt'):
        continue
    print(f'Rebuilding {filename}')
    data = torch.load(os.path.join(input_dir, filename), map_location='cpu', weights_only=False)
    sd = data['state_dict']
    rigid = data['rigid']
    ico_level = data['ico_level']

    model = GatUNet(
        in_channels=20,
        out_channels=3,
        num_heads=8,
        dropout=0.0,
        use_position_decoding=True,
        use_residual=False,
        ico_level=ico_level,
        input_dropout=0,
        euler_scale=None,
        rigid=rigid
    )

    model.load_state_dict(sd)
    out_name = filename.replace('.pt', '.model')
    out_subdir = ico_level
    os.makedirs(os.path.join(output_dir, out_subdir), exist_ok=True)
    out_path = os.path.join(output_dir, out_subdir, out_name)
    torch.save({'model': model}, out_path)
    print(f'Saved to {out_path}')

print('All done!')
