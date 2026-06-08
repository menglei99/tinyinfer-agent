# Docker 编译器后端

这台机器（任何没装 MSVC / g++ / clang 的 Windows 机器）跑不了 `cmake configure` —— 没编译器。本仓库的解决办法：把 `cmake / ctest` 跑在一个一次性 docker 容器里，`tinyinfer/` 通过 bind mount 进容器，build artifacts 也回写到宿主侧 `tinyinfer/build-docker/`。

切换由 `agent/tools/cmake_driver.py` 透明完成 —— **MCP server、LangGraph 节点都不知道**容器存在；任何 MCP 客户端（包括 Claude Desktop）调 `cmake_configure` 等工具时直接受益。

## 一次性准备

```bash
docker build -t tinyinfer-builder docker/builder/
```

镜像 base 是 `gcc:13`（约 ~1 GB），第一次拉 + apt-get 装 cmake/git 大约 3-5 min。

镜像里有：
- `g++` 13 / `gcc` 13
- `cmake` 3.25+（满足 tinyinfer CMakeLists 的 3.16 floor）
- `git`（FetchContent 拉 googletest 用）

## 启用

在仓库根创建 `.env`（参考 `.env.example`），加：

```
TINYINFER_DOCKER_IMAGE=tinyinfer-builder
TINYINFER_BUILD_DIR=tinyinfer/build-docker
```

或临时一次：

```bash
TINYINFER_DOCKER_IMAGE=tinyinfer-builder \
TINYINFER_BUILD_DIR=tinyinfer/build-docker \
.venv/Scripts/python.exe -m agent.cli analyze \
    --diff demo/diffs/sample_matmul.diff --mock --execute
```

期望终端 report：

```
## Execution (via MCP toolchain)
- `<ctest>` — PASS, compiled=True, ran=True
```

## 实际跑的命令

设上 env 后，`cmake_driver.configure(...)` 会把 host 路径翻译成 `/work/<rel>` 然后调：

```bash
docker run --rm \
    -v D:/tinyinfer-agent/tinyinfer:/work \
    -w /work \
    tinyinfer-builder \
    cmake -S /work -B /work/build-docker -DCMAKE_BUILD_TYPE=Release
```

`build()` 和 `ctest()` 同理（命令不一样，但容器调用形状一致）。每次都是 `--rm` 的一次性容器；没有长期容器需要管理。

## 为什么这么设计

- **切换点只有一个**：`cmake_driver` 内部的 `_docker_image()` 检测；上层（MCP server / LangGraph）零修改。
- **build dir 分流**：宿主 MSVC build 会走 `tinyinfer/build/`，docker build 走 `tinyinfer/build-docker/`，cmake cache 不混淆。
- **测试不依赖真 docker**：`tests/test_cmake_driver_docker.py` mock subprocess.run；真容器跑通的验证写在本文档底下。
- **挂载范围最小**：只挂 `tinyinfer/`，容器看不见 `agent/`、`.venv/` 等。

## 验证步骤

1. **构建 image**

    ```bash
    docker build -t tinyinfer-builder docker/builder/
    ```

2. **手动 sanity（最快验证容器自己 OK）**

    ```bash
    docker run --rm -v D:/tinyinfer-agent/tinyinfer:/work -w /work tinyinfer-builder \
        bash -c "cmake -S /work -B /work/build-docker -DCMAKE_BUILD_TYPE=Release && \
                 cmake --build /work/build-docker && \
                 ctest --test-dir /work/build-docker --output-on-failure"
    ```

3. **跑 pytest**

    ```bash
    .venv/Scripts/python.exe -m pytest tests/test_cmake_driver_docker.py -v
    ```

4. **端到端**：见上面"启用"段落。期望 ctest 一行 PASS。

## 已知坑

- **首次 FetchContent 慢**：cmake 第一次 configure 会克隆 googletest（~10–30 s），artifacts 落在 `tinyinfer/build-docker/_deps/`，二次 configure 复用。
- **Linux / Windows EOL**：生成的 `.cpp` 文件在 Windows 上以 CRLF 落盘也没事，g++ 兼容。如果遇到 preprocessor 报 stringification 问题再说。
- **stdout 编码**：`cmake_driver._run` 现在强制 `utf-8 / errors="replace"`，避免 Windows GBK 解码失败把整个 trace 炸掉。
