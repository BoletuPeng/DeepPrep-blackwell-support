# DeepPrep RTX 5090 (Blackwell Architecture) Installation Guide

## 📋 Who is this document for?

This guide is for you if:

- You have already pulled the Docker image `pbfslab/deepprep:25.1.0` following the official DeepPrep documentation
- You are using an **NVIDIA RTX 5090**, **RTX 5080**, or other **Blackwell architecture (sm_120)** GPU
- You encountered errors similar to those described below

### Error messages you may see

When you try to run DeepPrep, you may see warnings and errors like this:

```
NVIDIA GeForce RTX 5090 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 Laptop GPU GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**If you see the above error, this guide is for you.**

---

## 🔍 Root Cause Analysis

The official DeepPrep image (`pbfslab/deepprep:25.1.0`) has the following environment configuration:

| Component | Version | Issue |
|-----------|---------|-------|
| Base System | Ubuntu 22.04.5 LTS | No issue |
| PyTorch | **2.0.1+cu118** | Does not support Blackwell architecture (sm_120) |
| CUDA | **11.8** | Version too old |
| torch_geometric | 2.2.0 | Will cause compatibility issues after upgrade |

**Core Problem**: PyTorch 2.0.1 was compiled without support for Blackwell architecture. We need to upgrade to PyTorch 2.7.0+cu128, but this triggers a chain reaction requiring recompilation of multiple dependency packages.

---

## 🔧 Preparation

### Step 0.1: Choose a working directory

Choose a location with sufficient disk space as your working directory.

**Space requirements**:
- Downloaded files: ~6 GB
- Docker build process: Final image ~83 GB (original 45 GB + additional 38 GB)
- Recommended: **At least 100 GB free space**

Assume your chosen working directory is:
```
C:\Users\YourName\DeepPrep\
```

Create this folder in File Explorer.

---

### Step 0.2: Download PyTorch3D source code

Since PyTorch3D needs to be compiled from source, please download the source code in advance.

**Download URL**:
```
https://github.com/facebookresearch/pytorch3d/archive/refs/heads/main.zip
```

**Steps**:
1. Open the above link in your browser and download `pytorch3d-main.zip`
2. Extract the downloaded zip file to your working directory
3. Ensure the extracted folder is named `pytorch3d-main`

After completion, your working directory should look like this:
```
C:\Users\YourName\DeepPrep\
└── pytorch3d-main\
    ├── README.md
    ├── setup.py
    └── ...
```

---

### Step 0.3: Download CUDA Toolkit 12.8 offline installer

We need to install CUDA Toolkit inside the Docker container to compile CUDA code.

**How to choose the CUDA version?**

The CUDA version needs to match PyTorch's CUDA version. We will use PyTorch 2.7.0+**cu128**, so we need CUDA **12.8**.

**Download URL** (users in mainland China may prefer the .cn domain for faster speeds):
```
https://developer.download.nvidia.cn/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
```

Or the international site:
```
https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
```

**Note**: This is the **Linux version** of the installer (~5.4 GB), because we will use it inside the Docker container (Linux environment).

**Steps**:
1. Open the above link in your browser and download `cuda_12.8.0_570.86.10_linux.run`
2. Place the downloaded file directly in your working directory (no need to extract)

After completion, your working directory should look like this:
```
C:\Users\YourName\DeepPrep\
├── pytorch3d-main\
│   ├── README.md
│   └── ...
└── cuda_12.8.0_570.86.10_linux.run
```

---

### Step 0.4: Confirm the original Docker image is pulled

Open Command Prompt (press `Win+R`, type `cmd`, press Enter), and run the following command to confirm the original image exists:

```cmd
docker images pbfslab/deepprep:25.1.0
```

If you see output similar to this, the image exists:
```
REPOSITORY          TAG       IMAGE ID       CREATED       SIZE
pbfslab/deepprep    25.1.0    xxxxxxxxxxxx   x weeks ago   44.7GB
```

If not, please follow the official DeepPrep documentation to pull the image first.

---

## 🚀 Installation Steps

### Phase 1: Upgrade PyTorch

**Purpose**: Upgrade PyTorch from 2.0.1+cu118 to 2.7.0+cu128 to support Blackwell architecture (sm_120).

#### Step 1.1: Enter the working directory

Open a **new Command Prompt window** (press `Win+R`, type `cmd`, press Enter).

Replace the path in the following command with your actual working directory path, then paste it into Command Prompt and run:

```cmd
cd C:\Users\YourName\DeepPrep
```

**Verification**: After running, the current path in Command Prompt should display your working directory.

---

#### Step 1.2: Create PyTorch upgrade image

**Copy the following commands completely**, paste them into Command Prompt and run:

```cmd
echo FROM pbfslab/deepprep:25.1.0 > Dockerfile.step1
echo RUN /opt/conda/envs/deepprep/bin/pip uninstall -y torch torchvision torchaudio >> Dockerfile.step1
echo RUN /opt/conda/envs/deepprep/bin/pip install --no-cache-dir torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128 >> Dockerfile.step1

docker build -t deepprep:rtx5090-step1 -f Dockerfile.step1 .
```

**Estimated time**: 10-20 minutes (depending on network speed)

**What this step does**:
- Uninstalls old PyTorch (2.0.1+cu118)
- Installs new PyTorch (2.7.0+cu128), which supports sm_120 architecture

---

#### Step 1.3: Fix torch.load compatibility issue

**Copy the following commands completely**, paste them into Command Prompt and run:

```cmd
echo FROM deepprep:rtx5090-step1 > Dockerfile.step2
echo RUN find /opt/DeepPrep -name "*.py" -exec sed -i "s/torch\.load(\([^,)]*\), map_location=\([^,)]*\))/torch.load(\1, map_location=\2, weights_only=False)/g" {} \; >> Dockerfile.step2
echo RUN find /opt/DeepPrep -name "*.py" -exec sed -i "s/torch\.load(\([^,)]*\))/torch.load(\1, weights_only=False)/g" {} \; >> Dockerfile.step2

docker build -t deepprep:rtx5090-step2 -f Dockerfile.step2 .
```

**Estimated time**: 1-2 minutes

**What this step does**:

PyTorch 2.x enables `weights_only=True` by default for security reasons. However, DeepPrep's model files use pickle serialization and require `weights_only=False` to load properly. This step automatically modifies all relevant code.

---

### Phase 2: Recompile CUDA-related packages

**Purpose**: PyTorch3D, torch-scatter, and other packages contain CUDA code that must be recompiled for the new PyTorch version.

#### Step 2.1: Compile PyTorch3D and PyG-related packages

**Important**: Ensure you are still in your working directory (the directory containing the `pytorch3d-main` folder and `cuda_12.8.0_570.86.10_linux.run` file).

**Copy the following commands completely**, paste them into Command Prompt and run:

```cmd
echo FROM deepprep:rtx5090-step2 > Dockerfile.step3
echo RUN apt-get update ^&^& apt-get install -y build-essential ninja-build ^&^& rm -rf /var/lib/apt/lists/* >> Dockerfile.step3
echo COPY cuda_12.8.0_570.86.10_linux.run /tmp/cuda.run >> Dockerfile.step3
echo RUN chmod +x /tmp/cuda.run ^&^& /tmp/cuda.run --toolkit --silent ^&^& rm /tmp/cuda.run >> Dockerfile.step3
echo ENV PATH=/usr/local/cuda-12.8/bin:$PATH >> Dockerfile.step3
echo ENV CUDA_HOME=/usr/local/cuda-12.8 >> Dockerfile.step3
echo RUN /opt/conda/envs/deepprep/bin/pip uninstall -y pytorch3d >> Dockerfile.step3
echo COPY pytorch3d-main /tmp/pytorch3d >> Dockerfile.step3
echo ENV FORCE_CUDA=1 >> Dockerfile.step3
echo ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0" >> Dockerfile.step3
echo RUN cd /tmp/pytorch3d ^&^& /opt/conda/envs/deepprep/bin/pip install --no-cache-dir . >> Dockerfile.step3
echo RUN /opt/conda/envs/deepprep/bin/pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv >> Dockerfile.step3
echo RUN /opt/conda/envs/deepprep/bin/pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html >> Dockerfile.step3

docker build -t deepprep:rtx5090-step3 -f Dockerfile.step3 .
```

**Estimated time**: 30-60 minutes (PyTorch3D compilation is slow)

**What this step does**:
- Installs C++ compiler (build-essential) and build tools (ninja-build)
- Installs CUDA Toolkit 12.8 (provides nvcc compiler)
- Compiles PyTorch3D from source
- Installs torch-scatter, torch-sparse, and other packages matching PyTorch 2.7+cu128

**About `TORCH_CUDA_ARCH_LIST`**:
- `8.0` = Ampere (A100)
- `8.6` = Ampere (RTX 30 series)
- `9.0` = Hopper (H100)
- `12.0` = Blackwell (RTX 50 series) ← This is what we need

---

#### Step 2.2: Update pyg-lib

**Copy the following commands completely**, paste them into Command Prompt and run:

```cmd
echo FROM deepprep:rtx5090-step3 > Dockerfile.step4
echo RUN /opt/conda/envs/deepprep/bin/pip uninstall -y pyg-lib >> Dockerfile.step4
echo RUN /opt/conda/envs/deepprep/bin/pip install --no-cache-dir pyg-lib -f https://data.pyg.org/whl/torch-2.7.0+cu128.html >> Dockerfile.step4

docker build -t deepprep:rtx5090-step4 -f Dockerfile.step4 .
```

**Estimated time**: 2-5 minutes

**What this step does**:

pyg-lib also needs to match the new PyTorch version, otherwise you will encounter undefined symbol errors.

---

### Phase 3: Convert model files

**Purpose**: After torch_geometric upgrades from 2.2.0 to 2.7.0, the model serialization format has changed. We need to use the old environment to extract model parameters, then use the new environment to re-save them.

#### Step 3.1: Obtain model conversion scripts

**Method A (Recommended)**: If you cloned this repository from GitHub, the scripts are already in the `scripts/` folder. Simply copy them to your working directory:

```cmd
copy scripts\extract_state_dicts.py .
copy scripts\rebuild_models.py .
```

**Method B**: Manually create the script files. Use Notepad (or another text editor) to create `extract_state_dicts.py` with the following content:

<details>
<summary>Click to expand script content</summary>

```python
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
```

</details>

Save the file to your working directory with the filename `extract_state_dicts.py`.

After completion, your working directory should look like this:
```
C:\Users\YourName\DeepPrep\
├── pytorch3d-main\
├── cuda_12.8.0_570.86.10_linux.run
└── extract_state_dicts.py          ← Newly created file
```

---

#### Step 3.2: Use the original container to extract model parameters

**Copy the following commands completely**, paste them into Command Prompt and run:

```cmd
mkdir converted_models

docker run --rm -v %cd%:/scripts -v %cd%\converted_models:/output --entrypoint python pbfslab/deepprep:25.1.0 /scripts/extract_state_dicts.py
```

**Estimated time**: 1-2 minutes

**What this step does**:

Uses the **original container** (torch_geometric 2.2.0) to load model files and extract pure model parameters (state_dict). This avoids serialization format compatibility issues.

After completion, your working directory should look like this:
```
C:\Users\YourName\DeepPrep\
├── pytorch3d-main\
├── cuda_12.8.0_570.86.10_linux.run
├── extract_state_dicts.py
└── converted_models\               ← Newly created folder
    └── state_dicts\
        ├── lh_NoRigid_904_fsaverage3.pt
        ├── lh_NoRigid_904_fsaverage4.pt
        └── ... (10 .pt files in total)
```

---

#### Step 3.3: Prepare the model rebuild script

**Method A (Recommended)**: If you have already cloned this repository from GitHub, `rebuild_models.py` is already in your working directory.

**Method B**: Manually create the script file. Use Notepad to create `rebuild_models.py` with the following content:

<details>
<summary>Click to expand script content</summary>

```python
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
```

</details>

Save the file to your working directory with the filename `rebuild_models.py`.

---

#### Step 3.4: Use the new container to rebuild models

**Copy the following commands completely**, paste them into Command Prompt and run:

```cmd
mkdir new_models

docker run --rm --gpus all -v %cd%:/scripts -v %cd%\converted_models:/input -v %cd%\new_models:/output --entrypoint /opt/conda/envs/deepprep/bin/python deepprep:rtx5090-step4 /scripts/rebuild_models.py
```

**Estimated time**: 1-2 minutes

**What this step does**:

Uses the **new container** (torch_geometric 2.7.0) to re-instantiate the model class, load the previously extracted parameters, and save them. The resulting model files are fully compatible with the new torch_geometric version.

After completion, your working directory should look like this:
```
C:\Users\YourName\DeepPrep\
├── pytorch3d-main\
├── cuda_12.8.0_570.86.10_linux.run
├── extract_state_dicts.py
├── rebuild_models.py
├── converted_models\
│   └── state_dicts\
└── new_models\                     ← Newly created folder
    ├── fsaverage3\
    │   ├── lh_NoRigid_904_fsaverage3.model
    │   └── rh_NoRigid_904_fsaverage3.model
    ├── fsaverage4\
    ├── fsaverage5\
    └── fsaverage6\
```

---

### Phase 4: Build the final image

**Copy the following commands completely**, paste them into Command Prompt and run:

```cmd
echo FROM deepprep:rtx5090-step4 > Dockerfile.final
echo COPY new_models/fsaverage3 /opt/model/SUGAR/model_files/fsaverage3/ >> Dockerfile.final
echo COPY new_models/fsaverage4 /opt/model/SUGAR/model_files/fsaverage4/ >> Dockerfile.final
echo COPY new_models/fsaverage5 /opt/model/SUGAR/model_files/fsaverage5/ >> Dockerfile.final
echo COPY new_models/fsaverage6 /opt/model/SUGAR/model_files/fsaverage6/ >> Dockerfile.final

docker build -t deepprep:25.1.0-rtx5090 -f Dockerfile.final .
```

**Estimated time**: 1-2 minutes

**Congratulations! You now have a DeepPrep image that supports RTX 5090: `deepprep:25.1.0-rtx5090`**

---

## ✅ Verify Installation

Run the following command to verify the environment configuration is correct:

```cmd
docker run --rm --entrypoint bash deepprep:25.1.0-rtx5090 -c "/opt/conda/envs/deepprep/bin/pip list | grep -iE 'torch|pyg|scatter|sparse|cluster|spline|geometric|pytorch3d'"
```

**Expected output** should be similar to:

```
pyg-lib                      0.5.0+pt27cu128
pytorch3d                    0.7.9
torch                        2.7.0+cu128
torch_cluster                1.6.3+pt27cu128
torch-geometric              2.7.0
torch_scatter                2.1.2+pt27cu128
torch_sparse                 0.6.18+pt27cu128
torch_spline_conv            1.2.2+pt27cu128
torchaudio                   2.7.0+cu128
torchvision                  0.22.0+cu128
```

Key checkpoints:
- `torch` version should be `2.7.0+cu128`
- All `torch_*` packages should have the `pt27cu128` suffix
- `pyg-lib` should be `0.5.0+pt27cu128`

---

## 🎯 Run DeepPrep

Use the following command to run DeepPrep. Replace the paths with your actual paths:

```cmd
docker run -it --rm --gpus all ^
  -v D:\path\to\your\bids_dataset:/input ^
  -v D:\path\to\your\output:/output ^
  -v C:\path\to\your\license.txt:/fs_license.txt ^
  deepprep:25.1.0-rtx5090 ^
  /input /output participant ^
  --bold_task_type rest ^
  --fs_license_file /fs_license.txt ^
  --device 0
```

**Parameter explanation**:
- `-v D:\...\bids_dataset:/input`: Your BIDS format input data directory
- `-v D:\...\output:/output`: Output directory
- `-v C:\...\license.txt:/fs_license.txt`: FreeSurfer license file
- `--bold_task_type rest`: Task type (adjust according to your data)
- `--device 0`: Use the first GPU

To resume from an interruption, add the `--resume` parameter.

---

## 🧹 Cleanup (Optional)

After installation is complete and you have confirmed everything works correctly, you can delete intermediate files to save space.

### Delete temporary Dockerfiles

```cmd
del Dockerfile.step1 Dockerfile.step2 Dockerfile.step3 Dockerfile.step4 Dockerfile.final
```

### Delete intermediate Docker images

```cmd
docker rmi deepprep:rtx5090-step1 deepprep:rtx5090-step2 deepprep:rtx5090-step3 deepprep:rtx5090-step4
```

### Delete temporary folders (optional)

If you don't need to keep these files, you can manually delete:
- `converted_models` folder
- `new_models` folder
- `pytorch3d-main` folder
- `cuda_12.8.0_570.86.10_linux.run` file
- `extract_state_dicts.py` file
- `rebuild_models.py` file

---

## ❓ FAQ

### Q: Error "g++: command not found" when compiling PyTorch3D

**Cause**: C++ compiler not installed.

**Solution**: Ensure Dockerfile.step3 includes the `apt-get install -y build-essential` step.

---

### Q: Error "nvcc: command not found" when compiling

**Cause**: CUDA Toolkit not properly installed.

**Solution**:
1. Confirm `cuda_12.8.0_570.86.10_linux.run` file exists in your working directory
2. Confirm the filename is spelled correctly
3. Confirm you are running the docker build command in the correct directory

---

### Q: Still getting "no kernel image is available" error at runtime

**Cause**: PyTorch is still the old version.

**Solution**: Run the verification command to check the torch version, confirm it is `2.7.0+cu128`. If not, start again from step 1.2.

---

### Q: Model loading error containing "_lazy_load_hook" or similar

**Cause**: Model files are incompatible with the new torch_geometric version.

**Solution**: Please complete all steps in Phase 3 to ensure model files have been correctly converted.

---

### Q: Error "COPY failed: file not found" during Docker build

**Cause**: Docker cannot find the file to copy.

**Solution**:
1. Confirm you are running the command in the correct working directory
2. Confirm required files (such as `pytorch3d-main` folder, `cuda_12.8.0_570.86.10_linux.run` file) exist in the current directory
3. Use the `dir` command to view the current directory contents

---

## 📝 Technical Summary

Overview of compatibility issues resolved by this guide:

| Component | Original Version | New Version | Issue Description |
|-----------|------------------|-------------|-------------------|
| PyTorch | 2.0.1+cu118 | 2.7.0+cu128 | Old version doesn't support Blackwell (sm_120) |
| CUDA | 11.8 | 12.8 | Needs to match PyTorch cu128 |
| torch.load | weights_only=True | weights_only=False | Model loading failure |
| pytorch3d | Pre-compiled | Source compiled | CUDA/PyTorch version mismatch |
| torch-scatter etc. | pt20cu118 | pt27cu128 | ABI incompatibility |
| pyg-lib | Old version | 0.5.0+pt27cu128 | Undefined symbol error |
| torch_geometric | 2.2.0 | 2.7.0 | Model serialization format change |
| SUGAR models | Old format | New format | Need re-serialization |

---

*This guide was tested with DeepPrep 25.1.0 and NVIDIA GeForce RTX 5090 Laptop GPU.*

*If you have any questions or suggestions, please feel free to provide feedback.*
