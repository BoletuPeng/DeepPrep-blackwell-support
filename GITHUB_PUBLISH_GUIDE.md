# GitHub 发布指引

## 仓库结构预览

```
deepprep-blackwell-support/
├── README.md              # 仓库首页（双语）
├── INSTALL_GUIDE.md       # 详细安装指南
├── LICENSE                # MIT 许可证
├── .gitignore             # Git 忽略规则
└── scripts/
    ├── extract_state_dicts.py   # 模型提取脚本
    └── rebuild_models.py        # 模型重建脚本
```

---

## 发布步骤

### 方法一：通过 GitHub 网页界面（推荐新手）

#### 1. 创建仓库

1. 打开 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `deepprep-blackwell-support`
   - **Description**: `Guide for running DeepPrep on NVIDIA RTX 50 series (Blackwell) GPUs`
   - **Public** / Private：选择 Public（公开）以便他人使用
   - **不要**勾选 "Add a README file"（我们已经有了）
   - **不要**勾选 "Add .gitignore"（我们已经有了）
   - **不要**选择 License（我们已经有了）
3. 点击 **Create repository**

#### 2. 上传文件

创建仓库后，GitHub 会显示一个空仓库页面。

**方法 A：拖拽上传**

1. 点击 "uploading an existing file" 链接
2. 将以下文件拖拽到上传区域：
   - `README.md`
   - `INSTALL_GUIDE.md`
   - `LICENSE`
   - `.gitignore`
3. 点击 **Commit changes**

4. 创建 scripts 文件夹：
   - 点击 **Add file** → **Create new file**
   - 在文件名框中输入：`scripts/extract_state_dicts.py`
   - 粘贴脚本内容
   - 点击 **Commit changes**
   
5. 重复上一步，添加 `scripts/rebuild_models.py`

---

### 方法二：通过 Git 命令行

#### 1. 安装 Git

如果尚未安装 Git，请从 https://git-scm.com/download/win 下载安装。

#### 2. 配置 Git（首次使用）

打开命令提示符或 Git Bash，运行：

```bash
git config --global user.name "您的用户名"
git config --global user.email "您的邮箱"
```

#### 3. 在 GitHub 创建空仓库

1. 打开 https://github.com/new
2. **Repository name**: `deepprep-blackwell-support`
3. **不要**勾选任何初始化选项
4. 点击 **Create repository**
5. 记下仓库地址，例如：`https://github.com/YourUsername/deepprep-blackwell-support.git`

#### 4. 本地初始化并推送

假设您已将所有文件放在 `C:\Users\YourName\deepprep-blackwell-support\` 文件夹中：

```cmd
cd C:\Users\YourName\deepprep-blackwell-support

git init
git add .
git commit -m "Initial commit: DeepPrep Blackwell support guide"
git branch -M main
git remote add origin https://github.com/YourUsername/deepprep-blackwell-support.git
git push -u origin main
```

系统会提示您登录 GitHub（首次推送时）。

---

## 发布后的优化

### 添加 Topics（标签）

在仓库页面，点击右侧的齿轮图标（About 旁边），添加以下 Topics：

- `deepprep`
- `neuroimaging`
- `fmri`
- `pytorch`
- `nvidia`
- `rtx-5090`
- `blackwell`
- `docker`

### 创建 Release（可选）

1. 在仓库页面，点击右侧的 **Releases**
2. 点击 **Create a new release**
3. **Tag version**: `v1.0.0`
4. **Release title**: `DeepPrep Blackwell Support v1.0.0`
5. **Description**: 
   ```
   Initial release supporting:
   - DeepPrep 25.1.0
   - NVIDIA RTX 50 series (Blackwell architecture, sm_120)
   - PyTorch 2.7.0+cu128
   - CUDA 12.8
   
   Tested on RTX 5090 Laptop GPU.
   ```
6. 点击 **Publish release**

---

## 宣传建议

发布后，您可以在以下地方分享：

1. **DeepPrep GitHub Issues** - 如果 DeepPrep 官方仓库有相关 Issue，可以回复提供链接
2. **神经影像社区论坛** - 如 NeuroStars (https://neurostars.org)
3. **相关 Reddit 社区** - 如 r/neuroimaging, r/fMRI
4. **Twitter/X** - 使用 #neuroimaging #DeepPrep #RTX5090 标签

---

## 后续维护

- 如果 DeepPrep 发布新版本，可能需要更新本指南
- 关注 PyTorch 和 CUDA 的版本更新
- 接受社区反馈，修复文档中的问题
