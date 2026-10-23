# 参与 Fluxway 开发

[English](CONTRIBUTING.md)

本文说明开发环境、本地检查和拉取请求要求。

## 开发环境

使用 Linux 和 Python 3.12 或 3.13，在独立虚拟环境中安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[jax,profiler,visualization]' --group test --group lint --group build
```

## 本地检查

从仓库根目录运行格式检查和可移植 CPU 测试：

```bash
bash scripts/format.sh --check
JAX_PLATFORMS=cpu python -m pytest -ra --strict-markers --strict-config \
  --ignore=tests/checkpointer/test_retention.py \
  --ignore=tests/example/test_sparse_checkpoint_e2e.py
```

需要应用格式修复时运行 `bash scripts/format.sh`。排除的两个文件依赖可选的原生 `sparse_embedding` 和 Lance 环境；相关功能变更还需在具备依赖的环境中验证。

打包检查使用新的临时目录，避免混入旧产物：

```bash
mkdir -p tmp
package_check_dir=$(mktemp -d "$PWD/tmp/package-check.XXXXXX")
python -m build --outdir "$package_check_dir/dist"
python -m twine check --strict "$package_check_dir"/dist/*
python scripts/distribution.py check "$package_check_dir/dist"
python -m pytest --confcutdir=tests/packaging tests/packaging
```

隔离安装和完整发布验证见[版本发布](docs/releasing_zh.md)。本地单个 Python 环境的成功不能代替持续集成的 Python 3.12/3.13 矩阵。

## 拉取请求

- 每次变更聚焦一个问题，说明触发条件、最终行为和验证结果。
- 行为变更提供适当的测试，并同步更新中英文文档。
- 破坏性 API 变更说明迁移步骤。
- 合并前，最终提交必须通过所有必需的持续集成检查。

## 文档要求

- 中文正文和图中说明优先使用中文，保留 API、类型、字段和命令原名；中英文覆盖相同契约。
- 复杂流程使用 Mermaid 时序图，标明并发、等待、取消、背压、提交和失败边界；静态关系使用表格或结构图。
- 区分调用返回、任务完成、接收准入和持久提交；按当前实现核对责任归属与消息顺序。
- 示例注明依赖、输入、预期结果和适用范围，本地协议检查不能作为分布式恢复或性能证据。
- 修改后检查链接、代码块与图示渲染。图示约定和流程入口见[文档总览](docs/README_zh.md)。

## 持续集成产物

[夜间构建](.github/workflows/nightly.yml)复用完整持续集成流程，将已验证的 wheel 和源码包保存在 `python-distributions` 工作流产物中 14 天。工作流进入 `main` 后可手动运行；修改 CI 或夜间构建工作流的拉取请求也会运行夜间构建验证。版本标签、可信发布权限和发布顺序见[版本发布](docs/releasing_zh.md)。
