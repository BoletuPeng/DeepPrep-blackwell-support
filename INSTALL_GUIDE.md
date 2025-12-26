# DeepPrep RTX 5090 (Blackwell 架构) 安装指南

## 📋 这份文档是给谁的？

本指南适用于以下情况：

- 您已按照 DeepPrep 官方文档拉取了 Docker 镜像 `pbfslab/deepprep:25.1.0`
- 您使用的是 **NVIDIA RTX 5090**、**RTX 5080** 或其他 **Blackwell 架构 (sm_120)** 显卡
- 运行时遇到了类似下面的错误

### 您可能看到的错误信息

当您尝试运行 DeepPrep 时，可能会看到类似这样的警告和错误：

```
NVIDIA GeForce RTX 5090 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 Laptop GPU GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**如果您看到了上述错误，本指南就是为您准备的。**

---

## 🔍 问题根源分析

DeepPrep 官方镜像 (`pbfslab/deepprep:25.1.0`) 的环境配置如下：

| 组件 | 版本 | 问题 |
|------|------|------|
| 基础系统 | Ubuntu 22.04.5 LTS | 无问题 |
| PyTorch | **2.0.1+cu118** | 不支持 Blackwell 架构 (sm_120) |
| CUDA | **11.8** | 版本过旧 |
| torch_geometric | 2.2.0 | 将在升级后产生兼容性问题 |

**核心问题**：PyTorch 2.0.1 编译时未包含对 Blackwell 架构的支持。我们需要升级到 PyTorch 2.7.0+cu128，但这会引发一系列连锁反应，需要重新编译多个依赖包。

---

## 🔧 准备工作

### 步骤 0.1：选择工作目录

选择一个磁盘空间充足的位置作为工作目录。

**空间需求**：
- 下载文件：约 6 GB
- Docker 构建过程：最终镜像约 83 GB（原版 45 GB + 新增 38 GB）
- 建议预留：**至少 100 GB 可用空间**

假设您选择的工作目录是：
```
C:\Users\YourName\DeepPrep\
```

请在文件资源管理器中创建这个文件夹。

---

### 步骤 0.2：下载 PyTorch3D 源码

由于 PyTorch3D 需要从源码编译，请预先下载源码包。

**下载地址**：
```
https://github.com/facebookresearch/pytorch3d/archive/refs/heads/main.zip
```

**操作步骤**：
1. 在浏览器中打开上述链接，下载 `pytorch3d-main.zip`
2. 将下载的 zip 文件解压到您的工作目录
3. 确保解压后的文件夹名为 `pytorch3d-main`

完成后，您的工作目录应该是这样的：
```
C:\Users\YourName\DeepPrep\
└── pytorch3d-main\
    ├── README.md
    ├── setup.py
    └── ...
```

---

### 步骤 0.3：下载 CUDA Toolkit 12.8 离线安装包

我们需要在 Docker 容器内安装 CUDA Toolkit 以编译 CUDA 代码。

**如何选择 CUDA 版本？**

CUDA 版本需要与 PyTorch 的 CUDA 版本匹配。我们将使用 PyTorch 2.7.0+**cu128**，所以需要 CUDA **12.8**。

**下载地址**（中国大陆用户推荐使用 .cn 域名，速度更快）：
```
https://developer.download.nvidia.cn/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
```

或者国际站点：
```
https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
```

**注意**：这是 **Linux 版本**的安装包（约 5.4 GB），因为我们要在 Docker 容器（Linux 环境）内使用它。

**操作步骤**：
1. 在浏览器中打开上述链接，下载 `cuda_12.8.0_570.86.10_linux.run`
2. 将下载的文件直接放到您的工作目录（不需要解压）

完成后，您的工作目录应该是这样的：
```
C:\Users\YourName\DeepPrep\
├── pytorch3d-main\
│   ├── README.md
│   └── ...
└── cuda_12.8.0_570.86.10_linux.run
```

---

### 步骤 0.4：确认原版 Docker 镜像已拉取

打开命令提示符（按 `Win+R`，输入 `cmd`，回车），运行以下命令确认原版镜像存在：

```cmd
docker images pbfslab/deepprep:25.1.0
```

如果看到类似这样的输出，说明镜像已存在：
```
REPOSITORY          TAG       IMAGE ID       CREATED       SIZE
pbfslab/deepprep    25.1.0    xxxxxxxxxxxx   x weeks ago   44.7GB
```

如果没有，请先按照 DeepPrep 官方文档拉取镜像。

---

## 🚀 安装步骤

### 第一阶段：升级 PyTorch

**目的**：将 PyTorch 从 2.0.1+cu118 升级到 2.7.0+cu128，使其支持 Blackwell 架构 (sm_120)。

#### 步骤 1.1：进入工作目录

打开一个**新的命令提示符窗口**（按 `Win+R`，输入 `cmd`，回车）。

将下面的命令中的路径替换为您的实际工作目录路径，然后粘贴到命令提示符中运行：

```cmd
cd C:\Users\YourName\DeepPrep
```

**验证**：运行后，命令提示符的当前路径应该显示为您的工作目录。

---

#### 步骤 1.2：创建 PyTorch 升级镜像

将以下命令**完整复制**，粘贴到命令提示符中运行：

```cmd
echo FROM pbfslab/deepprep:25.1.0 > Dockerfile.step1
echo RUN /opt/conda/envs/deepprep/bin/pip uninstall -y torch torchvision torchaudio >> Dockerfile.step1
echo RUN /opt/conda/envs/deepprep/bin/pip install --no-cache-dir torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128 >> Dockerfile.step1

docker build -t deepprep:rtx5090-step1 -f Dockerfile.step1 .
```

**预计耗时**：10-20 分钟（取决于网络速度）

**这一步做了什么？**
- 卸载旧版 PyTorch (2.0.1+cu118)
- 安装新版 PyTorch (2.7.0+cu128)，该版本支持 sm_120 架构

---

#### 步骤 1.3：修复 torch.load 兼容性问题

将以下命令**完整复制**，粘贴到命令提示符中运行：

```cmd
echo FROM deepprep:rtx5090-step1 > Dockerfile.step2
echo RUN find /opt/DeepPrep -name "*.py" -exec sed -i "s/torch\.load(\([^,)]*\), map_location=\([^,)]*\))/torch.load(\1, map_location=\2, weights_only=False)/g" {} \; >> Dockerfile.step2
echo RUN find /opt/DeepPrep -name "*.py" -exec sed -i "s/torch\.load(\([^,)]*\))/torch.load(\1, weights_only=False)/g" {} \; >> Dockerfile.step2

docker build -t deepprep:rtx5090-step2 -f Dockerfile.step2 .
```

**预计耗时**：1-2 分钟

**这一步做了什么？**

PyTorch 2.x 出于安全考虑，默认启用了 `weights_only=True`。但 DeepPrep 的模型文件使用了 pickle 序列化，需要 `weights_only=False` 才能正常加载。这一步自动修改所有相关代码。

---

### 第二阶段：重新编译 CUDA 相关包

**目的**：PyTorch3D、torch-scatter 等包包含 CUDA 代码，升级 PyTorch 后必须针对新版本重新编译。

#### 步骤 2.1：编译 PyTorch3D 和 PyG 相关包

**重要**：请确保您仍在工作目录中（包含 `pytorch3d-main` 文件夹和 `cuda_12.8.0_570.86.10_linux.run` 文件的目录）。

将以下命令**完整复制**，粘贴到命令提示符中运行：

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

**预计耗时**：30-60 分钟（PyTorch3D 编译较慢）

**这一步做了什么？**
- 安装 C++ 编译器 (build-essential) 和构建工具 (ninja-build)
- 安装 CUDA Toolkit 12.8（提供 nvcc 编译器）
- 从源码编译 PyTorch3D
- 安装与 PyTorch 2.7+cu128 匹配的 torch-scatter、torch-sparse 等包

**关于 `TORCH_CUDA_ARCH_LIST`**：
- `8.0` = Ampere (A100)
- `8.6` = Ampere (RTX 30 系列)
- `9.0` = Hopper (H100)
- `12.0` = Blackwell (RTX 50 系列) ← 这是我们需要的

---

#### 步骤 2.2：更新 pyg-lib

将以下命令**完整复制**，粘贴到命令提示符中运行：

```cmd
echo FROM deepprep:rtx5090-step3 > Dockerfile.step4
echo RUN /opt/conda/envs/deepprep/bin/pip uninstall -y pyg-lib >> Dockerfile.step4
echo RUN /opt/conda/envs/deepprep/bin/pip install --no-cache-dir pyg-lib -f https://data.pyg.org/whl/torch-2.7.0+cu128.html >> Dockerfile.step4

docker build -t deepprep:rtx5090-step4 -f Dockerfile.step4 .
```

**预计耗时**：2-5 分钟

**这一步做了什么？**

pyg-lib 也需要匹配新的 PyTorch 版本，否则会出现符号未定义错误。

---

### 第三阶段：转换模型文件

**目的**：torch_geometric 从 2.2.0 升级到 2.7.0 后，模型序列化格式发生了变化。我们需要用旧版环境提取模型参数，再用新版环境重新保存。

#### 步骤 3.1：获取模型转换脚本

**方法 A（推荐）**：如果您是从 GitHub 克隆的本仓库，脚本已经在 `scripts/` 文件夹中，直接复制到工作目录即可：

```cmd
copy scripts\extract_state_dicts.py .
copy scripts\rebuild_models.py .
```

**方法 B**：手动创建脚本文件。使用记事本（或其他文本编辑器）创建 `extract_state_dicts.py`，内容如下：

<details>
<summary>点击展开脚本内容</summary>

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

将文件保存到您的工作目录，文件名为 `extract_state_dicts.py`。

完成后，您的工作目录应该是这样的：
```
C:\Users\YourName\DeepPrep\
├── pytorch3d-main\
├── cuda_12.8.0_570.86.10_linux.run
└── extract_state_dicts.py          ← 新创建的文件
```

---

#### 步骤 3.2：使用原版容器提取模型参数

将以下命令**完整复制**，粘贴到命令提示符中运行：

```cmd
mkdir converted_models

docker run --rm -v %cd%:/scripts -v %cd%\converted_models:/output --entrypoint python pbfslab/deepprep:25.1.0 /scripts/extract_state_dicts.py
```

**预计耗时**：1-2 分钟

**这一步做了什么？**

使用**原版容器**（torch_geometric 2.2.0）加载模型文件，提取出纯净的模型参数（state_dict）。这样可以避免序列化格式的兼容性问题。

完成后，您的工作目录应该是这样的：
```
C:\Users\YourName\DeepPrep\
├── pytorch3d-main\
├── cuda_12.8.0_570.86.10_linux.run
├── extract_state_dicts.py
└── converted_models\               ← 新创建的文件夹
    └── state_dicts\
        ├── lh_NoRigid_904_fsaverage3.pt
        ├── lh_NoRigid_904_fsaverage4.pt
        └── ... (共 10 个 .pt 文件)
```

---

#### 步骤 3.3：准备模型重建脚本

**方法 A（推荐）**：如果您已经从 GitHub 克隆了本仓库，`rebuild_models.py` 已经在工作目录中。

**方法 B**：手动创建脚本文件。使用记事本创建 `rebuild_models.py`，内容如下：

<details>
<summary>点击展开脚本内容</summary>

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

将文件保存到您的工作目录，文件名为 `rebuild_models.py`。

---

#### 步骤 3.4：使用新版容器重建模型

将以下命令**完整复制**，粘贴到命令提示符中运行：

```cmd
mkdir new_models

docker run --rm --gpus all -v %cd%:/scripts -v %cd%\converted_models:/input -v %cd%\new_models:/output --entrypoint /opt/conda/envs/deepprep/bin/python deepprep:rtx5090-step4 /scripts/rebuild_models.py
```

**预计耗时**：1-2 分钟

**这一步做了什么？**

使用**新版容器**（torch_geometric 2.7.0）重新实例化模型类，加载之前提取的参数，然后保存。这样生成的模型文件与新版 torch_geometric 完全兼容。

完成后，您的工作目录应该是这样的：
```
C:\Users\YourName\DeepPrep\
├── pytorch3d-main\
├── cuda_12.8.0_570.86.10_linux.run
├── extract_state_dicts.py
├── rebuild_models.py
├── converted_models\
│   └── state_dicts\
└── new_models\                     ← 新创建的文件夹
    ├── fsaverage3\
    │   ├── lh_NoRigid_904_fsaverage3.model
    │   └── rh_NoRigid_904_fsaverage3.model
    ├── fsaverage4\
    ├── fsaverage5\
    └── fsaverage6\
```

---

### 第四阶段：构建最终镜像

将以下命令**完整复制**，粘贴到命令提示符中运行：

```cmd
echo FROM deepprep:rtx5090-step4 > Dockerfile.final
echo COPY new_models/fsaverage3 /opt/model/SUGAR/model_files/fsaverage3/ >> Dockerfile.final
echo COPY new_models/fsaverage4 /opt/model/SUGAR/model_files/fsaverage4/ >> Dockerfile.final
echo COPY new_models/fsaverage5 /opt/model/SUGAR/model_files/fsaverage5/ >> Dockerfile.final
echo COPY new_models/fsaverage6 /opt/model/SUGAR/model_files/fsaverage6/ >> Dockerfile.final

docker build -t deepprep:25.1.0-rtx5090 -f Dockerfile.final .
```

**预计耗时**：1-2 分钟

**恭喜！您现在拥有了支持 RTX 5090 的 DeepPrep 镜像：`deepprep:25.1.0-rtx5090`**

---

## ✅ 验证安装

运行以下命令验证环境配置是否正确：

```cmd
docker run --rm --entrypoint bash deepprep:25.1.0-rtx5090 -c "/opt/conda/envs/deepprep/bin/pip list | grep -iE 'torch|pyg|scatter|sparse|cluster|spline|geometric|pytorch3d'"
```

**预期输出**应类似于：

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

关键检查点：
- `torch` 版本应为 `2.7.0+cu128`
- 所有 `torch_*` 包应带有 `pt27cu128` 后缀
- `pyg-lib` 应为 `0.5.0+pt27cu128`

---

## 🎯 运行 DeepPrep

使用以下命令运行 DeepPrep。请将路径替换为您的实际路径：

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

**参数说明**：
- `-v D:\...\bids_dataset:/input`：您的 BIDS 格式输入数据目录
- `-v D:\...\output:/output`：输出目录
- `-v C:\...\license.txt:/fs_license.txt`：FreeSurfer 许可证文件
- `--bold_task_type rest`：任务类型（根据您的数据调整）
- `--device 0`：使用第一个 GPU

如需从中断处继续运行，添加 `--resume` 参数。

---

## 🧹 清理工作（可选）

安装完成并确认一切正常后，您可以删除中间文件以节省空间。

### 删除临时 Dockerfile

```cmd
del Dockerfile.step1 Dockerfile.step2 Dockerfile.step3 Dockerfile.step4 Dockerfile.final
```

### 删除中间 Docker 镜像

```cmd
docker rmi deepprep:rtx5090-step1 deepprep:rtx5090-step2 deepprep:rtx5090-step3 deepprep:rtx5090-step4
```

### 删除临时文件夹（可选）

如果您不需要保留这些文件，可以手动删除：
- `converted_models` 文件夹
- `new_models` 文件夹
- `pytorch3d-main` 文件夹
- `cuda_12.8.0_570.86.10_linux.run` 文件
- `extract_state_dicts.py` 文件
- `rebuild_models.py` 文件

---

## ❓ 常见问题

### Q: 编译 PyTorch3D 时报错 "g++: command not found"

**原因**：C++ 编译器未安装。

**解决**：确保 Dockerfile.step3 中包含 `apt-get install -y build-essential` 步骤。

---

### Q: 编译时报错 "nvcc: command not found"

**原因**：CUDA Toolkit 未正确安装。

**解决**：
1. 确认 `cuda_12.8.0_570.86.10_linux.run` 文件存在于工作目录中
2. 确认文件名拼写正确
3. 确认您在正确的目录中运行 docker build 命令

---

### Q: 运行时仍然报错 "no kernel image is available"

**原因**：PyTorch 仍然是旧版本。

**解决**：运行验证命令检查 torch 版本，确认为 `2.7.0+cu128`。如果不是，请从步骤 1.2 重新开始。

---

### Q: 模型加载时报错包含 "_lazy_load_hook" 或类似信息

**原因**：模型文件与新版 torch_geometric 不兼容。

**解决**：请完整执行第三阶段的所有步骤，确保模型文件已正确转换。

---

### Q: Docker 构建过程中报错 "COPY failed: file not found"

**原因**：Docker 找不到要复制的文件。

**解决**：
1. 确认您在正确的工作目录中运行命令
2. 确认所需文件（如 `pytorch3d-main` 文件夹、`cuda_12.8.0_570.86.10_linux.run` 文件）存在于当前目录
3. 使用 `dir` 命令查看当前目录内容

---

## 📝 技术总结

本指南解决的兼容性问题一览：

| 组件 | 原版本 | 新版本 | 问题描述 |
|------|--------|--------|----------|
| PyTorch | 2.0.1+cu118 | 2.7.0+cu128 | 旧版不支持 Blackwell (sm_120) |
| CUDA | 11.8 | 12.8 | 需要匹配 PyTorch cu128 |
| torch.load | weights_only=True | weights_only=False | 模型加载失败 |
| pytorch3d | 预编译 | 源码编译 | CUDA/PyTorch 版本不匹配 |
| torch-scatter 等 | pt20cu118 | pt27cu128 | ABI 不兼容 |
| pyg-lib | 旧版 | 0.5.0+pt27cu128 | 符号未定义错误 |
| torch_geometric | 2.2.0 | 2.7.0 | 模型序列化格式变化 |
| SUGAR 模型 | 旧格式 | 新格式 | 需要重新序列化 |

---

*本指南基于 DeepPrep 25.1.0 和 NVIDIA GeForce RTX 5090 Laptop GPU 测试通过。*

*如有问题或建议，欢迎反馈。*
