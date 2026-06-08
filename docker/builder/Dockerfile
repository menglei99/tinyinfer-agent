# tinyinfer-builder: minimal Linux C++17 toolchain for building tinyinfer + ctest.
#
# Build:
#     docker build -t tinyinfer-builder docker/builder/
#
# Use (driven by agent.tools.cmake_driver when TINYINFER_DOCKER_IMAGE is set):
#     docker run --rm -v D:/tinyinfer-agent/tinyinfer:/work -w /work \
#         tinyinfer-builder cmake -S /work -B /work/build-docker ...
#
# Why gcc:13: glibc + g++ ready, no extra setup. cmake from apt (3.25 on
# Debian 12) satisfies tinyinfer's 3.16 floor. git is required by the
# CMakeLists FetchContent that pulls googletest at configure time.
FROM gcc:13

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
