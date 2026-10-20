# Conda 路径设置与注意事项

> 适用环境：Windows 11，Anaconda 安装于 `D:\Anaconda`（全机安装，当前账户只读）。
> 日常操作命令见 [个人笔记中的 conda 常用指令](../../../../00_个人笔记/conda常用指令.md)。
> 2026-10-11 仅迁移到 Python 分类并修正相对链接；原配置、命令与本机备忘保留。

## 一、环境存放路径的设置

### 核心配置项

写在**用户级配置文件** `C:\Users\你的用户名\.condarc` 中（文件不存在会自动新建）：

```yaml
envs_dirs:                 # 环境的存放位置，第一项优先级最高
  - D:\anaconda_envs\envs  # 新建环境（-n）默认落在这里
  - C:\Users\你的用户名\.conda\envs
pkgs_dirs:                 # 包缓存位置（解压前的包，占空间大头）
  - D:\anaconda_envs\pkgs
```

等价的命令行写法（`--add` 会把路径加到列表最前 = 最高优先级）：

```bash
conda config --add envs_dirs D:\anaconda_envs\envs
conda config --add pkgs_dirs D:\anaconda_envs\pkgs
```

### 关键规则

1. **`envs_dirs` 的第一项**才是 `conda create -n` 新建环境的默认落点，其余项只作为查找已有环境的位置。
2. **base 环境的位置由安装目录决定**，改 `envs_dirs` 不影响 base。
3. conda 会自动**跳过没有写权限的目录**，把可写的目录顶到前面用。
4. 用户级 `.condarc` 只影响当前用户，不污染 Anaconda 安装目录下的系统级配置。

### 修改后如何验证

```bash
conda info                        # 查看 envs directories 第一项是否为新路径
conda create -n _test python=3.11 -y   # 试建一个环境
conda remove -n _test --all -y         # 确认位置无误后删除
```

## 二、路径注意事项

### 目录选择

- 存放路径**不要含中文和空格**，避免莫名其妙的构建/激活问题。
- 用一个**独立目录**（如 `D:\anaconda_envs\envs`），不要直接用 Anaconda 安装目录下的 `envs`——全机方式安装时该目录对普通账户只读。
- `pkgs_dirs` 别省略：它是缓存大头，往往比环境本身还占空间，指到空间充足的盘。

### 本机当前状况备忘

- Anaconda 本体在 `D:\Anaconda`，但为只读安装，conda 把新环境默认落到了 **`C:\Users\Guards\.conda\envs`**，包缓存同理可能占 C 盘。
- 解决办法就是按上文第一节配置 `envs_dirs` + `pkgs_dirs` 指向 D 盘的自建目录。
- 若想迁移已有环境：用 `conda create -n 新名 --clone 旧名` 到新位置后删除旧环境；**不要直接移动文件夹**（pip 入口脚本的绝对路径和硬链接会失效）。

### 其他

- PyCharm / VSCode 中手动指定的解释器路径示例：
  `D:\anaconda_envs\envs\环境名\python.exe`
- 激活脚本报错时，先在 CMD 中执行一次 `conda init cmd.exe`（或对应 shell）重启终端。
