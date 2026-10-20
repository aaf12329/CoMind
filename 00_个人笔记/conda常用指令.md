# Conda 常用指令

> 适用环境：Windows 11，Anaconda 安装于 `D:\Anaconda`（全机安装，当前账户只读）。
> 环境路径的配置与坑见《[conda 路径设置与注意事项](../01_AI项目协作参考/01_编程规范/语言规范/Python/conda路径注意事项.md)》。

## 环境管理

### 创建与删除

```bash
# 创建环境（-n 后跟环境名，可指定 Python 版本）
conda create -n 环境名 python=3.11

# 按路径创建环境（不改全局配置时的临时方案）
conda create -p D:\anaconda_envs\envs\环境名 python=3.11

# 删除环境
conda remove -n 环境名 --all

# 克隆环境（迁移环境的官方做法，不要直接剪切文件夹）
conda create -n 新名 --clone 旧名
```

### 激活与退出

```bash
conda activate 环境名          # 激活命名环境
conda activate D:\路径\环境名   # 激活按 -p 创建的环境（需写全路径）
conda deactivate               # 退出当前环境
```

### 查看与搜索

```bash
conda env list        # 列出所有环境（带 * 为当前环境）
conda list            # 查看当前环境已装的包
conda list -n 环境名   # 查看指定环境的包
conda search 包名      # 搜索包的可用版本
```

### 安装与卸载包

```bash
conda install 包名            # 从 conda 源安装
conda install 包名=版本号      # 指定版本
pip install 包名              # conda 源没有的包再用 pip（在已激活的环境内）
conda remove 包名             # 卸载包
```

### 导出与还原环境

```bash
conda env export > environment.yml         # 导出环境配置
conda env create -f environment.yml        # 按配置文件重建环境
```
