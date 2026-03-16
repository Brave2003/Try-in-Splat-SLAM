# 提交到自己的 GitHub 仓库

本仓库已整理完毕，可按以下步骤推送到你自己的 GitHub。

## 1. 确认本地状态

```bash
cd /path/to/try
git status
```

确保没有不该提交的大文件（`.gitignore` 已排除 `*.pth`、`*.log`、`output/`、`log/` 等）。

## 2. 添加你自己的远程仓库

在 GitHub 上新建一个空仓库（不要勾选 README/license），然后：

```bash
# 添加你的仓库为 remote（替换为你的用户名和仓库名）
git remote add myrepo https://github.com/你的用户名/你的仓库名.git

# 或使用 SSH
# git remote add myrepo git@github.com:你的用户名/你的仓库名.git
```

若想直接替换原来的 origin：

```bash
git remote set-url origin https://github.com/你的用户名/你的仓库名.git
```

## 3. 提交并推送

```bash
# 暂存所有更改（会遵守 .gitignore）
git add -A
git status   # 再检查一遍，确认没有大文件

# 提交
git commit -m "Initial commit: try project"

# 推送到你的仓库（若用 myrepo）
git push -u myrepo main

# 若已把 origin 改成你的仓库
# git push -u origin main
```

## 4. 子模块说明

本仓库包含子模块（如 `thirdparty/simple-knn`、`thirdparty/diff-gaussian-rasterization-w-pose` 等）。**lietorch** 已从 `.gitmodules` 中移除（避免本地子模块引用异常）；若需要，克隆后请手动执行：

```bash
git clone https://github.com/princeton-vl/lietorch.git thirdparty/lietorch
```

其他子模块：别人克隆时需使用：

```bash
git clone --recursive https://github.com/你的用户名/你的仓库名.git
```

若你本地子模块未初始化或报错，可执行：

```bash
git submodule update --init --recursive
```

## 5. 子模块报错时（如 lietorch）

若执行 `git status` 出现类似：

```text
fatal: not a git repository: thirdparty/lietorch/../../.git/modules/thirdparty/lietorch
```

说明本地子模块未正确初始化。可先尝试：

```bash
git submodule update --init thirdparty/lietorch
```

若因网络等原因失败，可暂时解除该子模块再推送（之后在新克隆的仓库里再执行 `git submodule update --init --recursive` 即可恢复）：

```bash
git submodule deinit -f thirdparty/lietorch
git rm --cached thirdparty/lietorch
# 若不想把 lietorch 内容提交，可加入 .gitignore：thirdparty/lietorch/
# 然后正常 add / commit / push
```

## 6. 安全提醒

- 已从 `origin` 的 URL 中移除了个人访问令牌（token），避免泄露。
- 推送时如需认证，请使用 SSH 密钥，或在本地使用 Git 凭据管理器，不要再把 token 写进仓库或配置文件。
