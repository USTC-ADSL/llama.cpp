#!/bin/bash

# Hexagon/OpenCL/QNN/Vulkan 编译脚本
# 用于编译支持 Snapdragon CPU、Hexagon、OpenCL、QNN 与 Vulkan 的 llama.cpp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="build"
PRESET="arm64-android-snapdragon-release"

resolve_first_existing_dir() {
    local candidate
    for candidate in "$@"; do
        if [ -n "$candidate" ] && [ -d "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

# 默认启用 Hexagon NPU 与 OpenCL，可通过参数覆盖
ENABLE_NPU=1
ENABLE_GPU=1
ENABLE_VULKAN=0
ENABLE_QNN=0
ENABLE_QNN_CPU_BACKEND=1
ENABLE_QNN_HEXAGON_BACKEND=0
ENABLE_PROFILING=0
VULKAN_GLSLC_EXECUTABLE=""
VULKAN_INCLUDE_DIR_OVERRIDE=""

ENV_QNN_SDK_PATH="${QNN_SDK_PATH:-}"
ENV_QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
QNN_SDK_PATH="$(resolve_first_existing_dir \
    "$ENV_QNN_SDK_PATH" \
    "$ENV_QNN_SDK_ROOT" \
    "$SCRIPT_DIR/../qairt/2.31.0.250130" \
    "$SCRIPT_DIR/../qairt" || true)"

# 让 ccache 使用仓库内可写目录，避免某些环境下 /run 目录无权限
export CCACHE_DIR="${CCACHE_DIR:-$SCRIPT_DIR/.ccache}"
export CCACHE_TEMPDIR="${CCACHE_TEMPDIR:-$CCACHE_DIR/tmp}"
mkdir -p "$CCACHE_TEMPDIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${BLUE}========================================${NC}\n${BLUE}$1${NC}\n${BLUE}========================================${NC}\n"; }

feature_status() {
    if [ "$1" -eq 1 ]; then
        echo "启用"
    else
        echo "禁用"
    fi
}

print_cli_help() {
    cat <<EOF
用法:
  $(basename "$0") [build_dir] [preset] [选项]

说明:
  CPU 后端始终会被编入，无需显式开关。

选项:
  --with-npu / --enable-npu                 启用 ggml-hexagon Hexagon NPU (默认)
  --without-npu / --no-npu                  禁用 ggml-hexagon Hexagon NPU
  --with-gpu / --enable-gpu                 启用 OpenCL GPU (默认)
  --without-gpu / --no-gpu                  禁用 OpenCL GPU
  --with-qnn / --enable-qnn                 启用 QNN 后端
  --without-qnn / --no-qnn                  禁用 QNN 后端 (默认)
  --qnn-sdk <path>                          指定 QNN SDK 根目录
  --with-qnn-cpu-backend                    启用 qnn-cpu 设备 (QNN 开启时默认)
  --without-qnn-cpu-backend                 禁用 qnn-cpu 设备
  --with-qnn-hexagon-backend                启用 QNN Hexagon custom package
  --without-qnn-hexagon-backend             禁用 QNN Hexagon custom package (默认)
  --with-vulkan / --enable-vulkan           启用 Vulkan GPU
  --without-vulkan / --no-vulkan            禁用 Vulkan GPU (默认)
  --vulkan-glslc <path>                     指定 glslc 可执行文件路径
  --vulkan-include <path>                   指定 Vulkan 头文件目录（需包含 vulkan/vulkan.hpp）
  --with-profiling / --profiling            启用 Stage Profiling (CPU + OpenCL + Vulkan)
  -h, --help                                显示本帮助

示例:
  $0 build-opencl arm64-android-snapdragon-release --without-npu --with-gpu
  $0 build-qnn arm64-android-snapdragon-release --without-npu --with-qnn --qnn-sdk /path/to/qairt
  $0 build-mixed arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn --qnn-sdk /path/to/qairt
  $0 build-vulkan arm64-android-snapdragon-release --with-vulkan --without-gpu --without-npu
EOF
}

parse_args() {
    local positional=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --with-npu|--enable-npu)
                ENABLE_NPU=1
                shift
                ;;
            --without-npu|--no-npu|--disable-npu)
                ENABLE_NPU=0
                shift
                ;;
            --with-gpu|--enable-gpu)
                ENABLE_GPU=1
                shift
                ;;
            --without-gpu|--no-gpu|--disable-gpu)
                ENABLE_GPU=0
                shift
                ;;
            --with-qnn|--enable-qnn)
                ENABLE_QNN=1
                shift
                ;;
            --without-qnn|--no-qnn|--disable-qnn)
                ENABLE_QNN=0
                shift
                ;;
            --qnn-sdk)
                QNN_SDK_PATH="$2"
                shift 2
                ;;
            --with-qnn-cpu-backend|--enable-qnn-cpu-backend)
                ENABLE_QNN=1
                ENABLE_QNN_CPU_BACKEND=1
                shift
                ;;
            --without-qnn-cpu-backend|--no-qnn-cpu-backend|--disable-qnn-cpu-backend)
                ENABLE_QNN_CPU_BACKEND=0
                shift
                ;;
            --with-qnn-hexagon-backend|--enable-qnn-hexagon-backend)
                ENABLE_QNN=1
                ENABLE_QNN_HEXAGON_BACKEND=1
                shift
                ;;
            --without-qnn-hexagon-backend|--no-qnn-hexagon-backend|--disable-qnn-hexagon-backend)
                ENABLE_QNN_HEXAGON_BACKEND=0
                shift
                ;;
            --with-vulkan|--enable-vulkan)
                ENABLE_VULKAN=1
                shift
                ;;
            --without-vulkan|--no-vulkan|--disable-vulkan)
                ENABLE_VULKAN=0
                shift
                ;;
            --vulkan-glslc)
                VULKAN_GLSLC_EXECUTABLE="$2"
                shift 2
                ;;
            --vulkan-include)
                VULKAN_INCLUDE_DIR_OVERRIDE="$2"
                shift 2
                ;;
            --with-profiling|--profiling|--enable-profiling)
                ENABLE_PROFILING=1
                shift
                ;;
            --without-profiling|--no-profiling|--disable-profiling)
                ENABLE_PROFILING=0
                shift
                ;;
            -h|--help)
                print_cli_help
                exit 0
                ;;
            --)
                shift
                while [[ $# -gt 0 ]]; do
                    positional+=("$1")
                    shift
                done
                break
                ;;
            *)
                positional+=("$1")
                shift
                ;;
        esac
    done

    if [ ${#positional[@]} -ge 1 ]; then
        BUILD_DIR="${positional[0]}"
    fi

    if [ ${#positional[@]} -ge 2 ]; then
        PRESET="${positional[1]}"
    fi
}

# 设置 Hexagon SDK（ggml-hexagon 或 QNN custom package 使用）
setup_hexagon_env() {
    log_section "设置 Hexagon SDK"

    export HEXAGON_SDK_ROOT="$SCRIPT_DIR/hexagon-sdk"
    export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04"

    log_info "HEXAGON_SDK_ROOT: $HEXAGON_SDK_ROOT"
    log_info "HEXAGON_TOOLS_ROOT: $HEXAGON_TOOLS_ROOT"

    if [ ! -f "$HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" ]; then
        log_error "Hexagon 工具未找到: $HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang"
        exit 1
    fi
    log_info "✓ Hexagon 工具已验证"
}

setup_qnn_env() {
    log_section "设置 QNN SDK"

    if [ -z "$QNN_SDK_PATH" ]; then
        log_error "未找到 QNN SDK，请通过 --qnn-sdk 指定，或设置 QNN_SDK_PATH/QNN_SDK_ROOT"
        exit 1
    fi

    if [ ! -d "$QNN_SDK_PATH" ]; then
        log_error "QNN SDK 目录不存在: $QNN_SDK_PATH"
        exit 1
    fi

    QNN_SDK_PATH="$(cd "$QNN_SDK_PATH" && pwd)"
    export QNN_SDK_PATH
    export QNN_SDK_ROOT="$QNN_SDK_PATH"

    log_info "QNN_SDK_PATH: $QNN_SDK_PATH"
    log_info "✓ QNN SDK 已验证"
}

setup_vulkan_env() {
    log_section "设置 Vulkan 工具链"

    local candidates=()
    if [ -n "${VULKAN_GLSLC_EXECUTABLE}" ]; then
        candidates+=("${VULKAN_GLSLC_EXECUTABLE}")
    fi

    if [ -n "${ANDROID_NDK_ROOT:-}" ]; then
        candidates+=("${ANDROID_NDK_ROOT}/shader-tools/linux-x86_64/glslc")
    fi

    candidates+=(
        "$SCRIPT_DIR/android-ndk-r27d/shader-tools/linux-x86_64/glslc"
        "$SCRIPT_DIR/hexagon-sdk/tools/android-ndk-r25c/shader-tools/linux-x86_64/glslc"
    )

    local found_glslc=""
    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -x "$candidate" ]; then
            found_glslc="$candidate"
            break
        fi
    done

    if [ -z "$found_glslc" ] && command -v glslc >/dev/null 2>&1; then
        found_glslc="$(command -v glslc)"
    fi

    if [ -z "$found_glslc" ]; then
        log_error "未找到 glslc；请安装 Vulkan shader tools 或通过 --vulkan-glslc 指定路径"
        exit 1
    fi

    local include_candidates=()
    if [ -n "${VULKAN_INCLUDE_DIR_OVERRIDE}" ]; then
        include_candidates+=("${VULKAN_INCLUDE_DIR_OVERRIDE}")
    fi
    include_candidates+=(
        "$SCRIPT_DIR/hexagon-sdk/tools/android-ndk-r25c/sources/third_party/vulkan/src/include"
        "$SCRIPT_DIR/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include"
    )

    local found_include=""
    for candidate in "${include_candidates[@]}"; do
        if [ -f "$candidate/vulkan/vulkan.hpp" ]; then
            found_include="$candidate"
            break
        fi
    done

    if [ -z "$found_include" ]; then
        log_error "未找到包含 vulkan/vulkan.hpp 的 Vulkan 头目录；请通过 --vulkan-include 指定路径"
        exit 1
    fi

    VULKAN_GLSLC_EXECUTABLE="$found_glslc"
    VULKAN_INCLUDE_DIR_OVERRIDE="$found_include"
    export PATH="$(dirname "$VULKAN_GLSLC_EXECUTABLE"):$PATH"

    log_info "Vulkan glslc: $VULKAN_GLSLC_EXECUTABLE"
    log_info "Vulkan includes: $VULKAN_INCLUDE_DIR_OVERRIDE"
    "$VULKAN_GLSLC_EXECUTABLE" --version | head -n 1 || true
}

configure_build() {
    log_section "配置编译"

    cd "$SCRIPT_DIR"

    local hexagon_flag="OFF"
    local opencl_flag="OFF"
    local vulkan_flag="OFF"
    local qnn_flag="OFF"
    local qnn_cpu_backend_flag="OFF"
    local qnn_hexagon_backend_flag="OFF"

    if [ "$ENABLE_NPU" -eq 1 ]; then
        hexagon_flag="ON"
    fi

    if [ "$ENABLE_GPU" -eq 1 ]; then
        opencl_flag="ON"
    fi

    if [ "$ENABLE_VULKAN" -eq 1 ]; then
        vulkan_flag="ON"
    fi

    if [ "$ENABLE_QNN" -eq 1 ]; then
        qnn_flag="ON"
        if [ "$ENABLE_QNN_CPU_BACKEND" -eq 1 ]; then
            qnn_cpu_backend_flag="ON"
        fi
        if [ "$ENABLE_QNN_HEXAGON_BACKEND" -eq 1 ]; then
            qnn_hexagon_backend_flag="ON"
        fi
    fi

    log_info "使用预设: $PRESET"
    log_info "构建目录: $BUILD_DIR"
    log_info "CPU: 始终启用"
    log_info "Hexagon NPU: $(feature_status $ENABLE_NPU)"
    log_info "OpenCL GPU: $(feature_status $ENABLE_GPU)"
    log_info "QNN: $(feature_status $ENABLE_QNN)"
    if [ "$ENABLE_QNN" -eq 1 ]; then
        log_info "QNN CPU backend: $(feature_status $ENABLE_QNN_CPU_BACKEND)"
        log_info "QNN Hexagon custom package: $(feature_status $ENABLE_QNN_HEXAGON_BACKEND)"
    fi
    log_info "Vulkan GPU: $(feature_status $ENABLE_VULKAN)"
    log_info "Stage Profiling: $(feature_status $ENABLE_PROFILING)"

    rm -rf "$BUILD_DIR"

    local cmake_args=(
        --preset "$PRESET"
        -B "$BUILD_DIR"
        -DGGML_HEXAGON="$hexagon_flag"
        -DGGML_OPENCL="$opencl_flag"
        -DGGML_VULKAN="$vulkan_flag"
    )

    if [ "$ENABLE_PROFILING" -eq 1 ]; then
        cmake_args+=(
            -DGGML_OPENCL_PROFILING=ON
            -DGGML_CPU_PROFILING=ON
            -DGGML_VULKAN_PROFILING=ON
        )
    fi

    if [ "$ENABLE_NPU" -eq 1 ] || [ "$ENABLE_QNN_HEXAGON_BACKEND" -eq 1 ]; then
        cmake_args+=(
            -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT"
            -DHEXAGON_TOOLS_ROOT="$HEXAGON_TOOLS_ROOT"
        )
    fi

    if [ "$ENABLE_QNN" -eq 1 ]; then
        cmake_args+=(
            -DGGML_QNN=ON
            -DGGML_QNN_SDK_PATH="$QNN_SDK_PATH"
            -DGGML_QNN_ENABLE_CPU_BACKEND="$qnn_cpu_backend_flag"
            -DGGML_QNN_ENABLE_HEXAGON_BACKEND="$qnn_hexagon_backend_flag"
        )
    fi

    if [ "$ENABLE_VULKAN" -eq 1 ]; then
        cmake_args+=(
            -DVulkan_GLSLC_EXECUTABLE="$VULKAN_GLSLC_EXECUTABLE"
            -DVulkan_INCLUDE_DIR="$VULKAN_INCLUDE_DIR_OVERRIDE"
        )
    fi

    cmake "${cmake_args[@]}"

    log_info "✓ 配置完成"
}

build() {
    log_section "构建"

    cd "$SCRIPT_DIR"

    local num_jobs
    num_jobs=$(nproc 2>/dev/null || echo 4)
    log_info "使用 $num_jobs 个并行任务"

    cmake --build "$BUILD_DIR" -j "$num_jobs"

    log_info "✓ 构建完成"
}

install_htp_libs() {
    log_section "安装 HTP 库"

    if [ "$ENABLE_NPU" -ne 1 ]; then
        log_info "已禁用 ggml-hexagon，跳过 HTP 库复制"
        return
    fi

    local hexagon_build_dir="$SCRIPT_DIR/$BUILD_DIR/ggml/src/ggml-hexagon"
    local bin_dir="$SCRIPT_DIR/$BUILD_DIR/bin"

    for version in v73 v75 v79 v81; do
        local htp_lib="libggml-htp-${version}.so"
        if [ -f "$hexagon_build_dir/$htp_lib" ]; then
            cp "$hexagon_build_dir/$htp_lib" "$bin_dir/"
            log_info "✓ 已复制 $htp_lib"
        else
            log_warn "✗ 未找到 $htp_lib"
        fi
    done

    log_info "✓ HTP 库安装完成"
}

verify_build() {
    log_section "验证输出"

    local exe="$SCRIPT_DIR/$BUILD_DIR/bin/llama-cli"
    local bin_dir="$SCRIPT_DIR/$BUILD_DIR/bin"

    if [ ! -f "$exe" ]; then
        log_error "可执行文件未生成: $exe"
        exit 1
    fi

    log_info "✓ 可执行文件已生成: $exe"
    log_info "文件大小: $(du -h "$exe" | cut -f1)"

    log_info "后端库:"

    if [ "$ENABLE_NPU" -eq 1 ]; then
        local hex_lib="libggml-hexagon.so"
        if [ -f "$bin_dir/$hex_lib" ]; then
            log_info "  ✓ $hex_lib ($(du -h "$bin_dir/$hex_lib" | cut -f1))"
        else
            log_warn "  ✗ $hex_lib 缺失"
        fi
    else
        log_info "  - ggml-hexagon 已禁用"
    fi

    if [ "$ENABLE_GPU" -eq 1 ]; then
        local ocl_lib="libggml-opencl.so"
        if [ -f "$bin_dir/$ocl_lib" ]; then
            log_info "  ✓ $ocl_lib ($(du -h "$bin_dir/$ocl_lib" | cut -f1))"
        else
            log_warn "  ✗ $ocl_lib 缺失"
        fi
    else
        log_info "  - OpenCL GPU 已禁用"
    fi

    if [ "$ENABLE_QNN" -eq 1 ]; then
        local qnn_lib="libggml-qnn.so"
        if [ -f "$bin_dir/$qnn_lib" ]; then
            log_info "  ✓ $qnn_lib ($(du -h "$bin_dir/$qnn_lib" | cut -f1))"
        else
            log_warn "  ✗ $qnn_lib 缺失"
        fi
    else
        log_info "  - QNN 已禁用"
    fi

    if [ "$ENABLE_VULKAN" -eq 1 ]; then
        local vk_lib="libggml-vulkan.so"
        if [ -f "$bin_dir/$vk_lib" ]; then
            log_info "  ✓ $vk_lib ($(du -h "$bin_dir/$vk_lib" | cut -f1))"
        else
            log_warn "  ✗ $vk_lib 缺失"
        fi
    else
        log_info "  - Vulkan GPU 已禁用"
    fi

    local cpu_lib="libggml-cpu.so"
    if [ -f "$bin_dir/$cpu_lib" ]; then
        log_info "  ✓ $cpu_lib ($(du -h "$bin_dir/$cpu_lib" | cut -f1))"
    else
        log_warn "  ✗ $cpu_lib 缺失"
    fi

    if [ "$ENABLE_NPU" -eq 1 ]; then
        log_info "HTP 库 (ggml-hexagon):"
        for version in v73 v75 v79 v81; do
            local htp_lib="libggml-htp-${version}.so"
            if [ -f "$bin_dir/$htp_lib" ]; then
                log_info "  ✓ $htp_lib ($(du -h "$bin_dir/$htp_lib" | cut -f1))"
            else
                log_warn "  ✗ $htp_lib"
            fi
        done
    fi

    if [ "$ENABLE_QNN" -eq 1 ]; then
        log_info "QNN 运行时库:"
        local qnn_runtime_lib
        for qnn_runtime_lib in libQnnSystem.so libQnnGpu.so libQnnHtp.so; do
            if [ -f "$bin_dir/$qnn_runtime_lib" ]; then
                log_info "  ✓ $qnn_runtime_lib ($(du -h "$bin_dir/$qnn_runtime_lib" | cut -f1))"
            else
                log_warn "  ✗ $qnn_runtime_lib"
            fi
        done
        if [ "$ENABLE_QNN_CPU_BACKEND" -eq 1 ]; then
            if [ -f "$bin_dir/libQnnCpu.so" ]; then
                log_info "  ✓ libQnnCpu.so ($(du -h "$bin_dir/libQnnCpu.so" | cut -f1))"
            else
                log_warn "  ✗ libQnnCpu.so"
            fi
            if [ -f "$bin_dir/libomp.so" ]; then
                log_info "  ✓ libomp.so ($(du -h "$bin_dir/libomp.so" | cut -f1))"
            else
                log_warn "  ✗ libomp.so"
            fi
        fi
    fi

    if command -v readelf >/dev/null 2>&1; then
        log_info "后端支持:"
        if [ "$ENABLE_NPU" -eq 1 ]; then
            readelf -d "$exe" 2>/dev/null | grep -q "libggml-hexagon.so" && log_info "  ✓ Hexagon NPU" || log_warn "  ✗ Hexagon NPU"
        else
            log_info "  - Hexagon NPU 已禁用"
        fi

        if [ "$ENABLE_GPU" -eq 1 ]; then
            readelf -d "$exe" 2>/dev/null | grep -q "libggml-opencl.so" && log_info "  ✓ OpenCL GPU" || log_warn "  ✗ OpenCL GPU"
        else
            log_info "  - OpenCL GPU 已禁用"
        fi

        if [ "$ENABLE_VULKAN" -eq 1 ]; then
            readelf -d "$exe" 2>/dev/null | grep -q "libggml-vulkan.so" && log_info "  ✓ Vulkan GPU" || log_warn "  ✗ Vulkan GPU"
        else
            log_info "  - Vulkan GPU 已禁用"
        fi

        if [ "$ENABLE_QNN" -eq 1 ]; then
            if [ -f "$bin_dir/libggml-qnn.so" ]; then
                log_info "  ✓ QNN backend library"
            else
                log_warn "  ✗ QNN backend library"
            fi
        else
            log_info "  - QNN 已禁用"
        fi

        readelf -d "$exe" 2>/dev/null | grep -q "libggml-cpu.so" && log_info "  ✓ CPU 后端" || log_warn "  ✗ CPU 后端"
    fi
}

show_usage() {
    log_section "编译完成"

    local exe="$SCRIPT_DIR/$BUILD_DIR/bin/llama-cli"
    local bench="$SCRIPT_DIR/$BUILD_DIR/bin/llama-bench"

    echo "构建参数:"
    echo "  CPU: 始终启用"
    echo "  Hexagon NPU: $(feature_status $ENABLE_NPU)"
    echo "  OpenCL GPU: $(feature_status $ENABLE_GPU)"
    echo "  QNN: $(feature_status $ENABLE_QNN)"
    if [ "$ENABLE_QNN" -eq 1 ]; then
        echo "  QNN CPU backend: $(feature_status $ENABLE_QNN_CPU_BACKEND)"
        echo "  QNN Hexagon custom package: $(feature_status $ENABLE_QNN_HEXAGON_BACKEND)"
    fi
    echo "  Vulkan GPU: $(feature_status $ENABLE_VULKAN)"
    echo "  Profiling: $(feature_status $ENABLE_PROFILING)"
    echo "  目录: $BUILD_DIR"
    echo ""

    echo "使用示例:"
    echo ""
    echo "  # 查看可用设备"
    echo "  $bench --list-devices"
    echo ""
    echo "  # 基本推理 (CPU)"
    echo "  $exe -m model.gguf -p \"Hello world\" -ngl 0"
    echo ""
    if [ "$ENABLE_GPU" -eq 1 ]; then
        echo "  # 使用 OpenCL"
        echo "  $exe -m model.gguf -p \"Hello world\" -ngl 99 -dev GPUOpenCL"
        echo ""
    fi
    if [ "$ENABLE_QNN" -eq 1 ]; then
        echo "  # 使用 QNN GPU"
        echo "  $exe -m model.gguf -p \"Hello world\" -ngl 99 -dev qnn-gpu"
        echo ""
        echo "  # 使用 QNN NPU"
        echo "  $exe -m model.gguf -p \"Hello world\" -ngl 99 -dev qnn-npu"
        echo ""
        if [ "$ENABLE_QNN_CPU_BACKEND" -eq 1 ]; then
            echo "  # 使用 QNN CPU"
            echo "  $exe -m model.gguf -p \"Hello world\" -ngl 99 -dev qnn-cpu"
            echo ""
        fi
    fi
    if [ "$ENABLE_NPU" -eq 1 ]; then
        echo "  # 使用 ggml-hexagon"
        echo "  $exe -m model.gguf -p \"Hello world\" --hexagon"
        echo ""
    fi
    if [ "$ENABLE_VULKAN" -eq 1 ]; then
        echo "  # 使用 Vulkan"
        echo "  $exe -m model.gguf -p \"Hello world\" -ngl 99 -dev Vulkan0"
        echo ""
    fi
}

main() {
    parse_args "$@"

    log_info "llama.cpp Hexagon/OpenCL/QNN/Vulkan 编译脚本"
    log_info "构建目录: $BUILD_DIR | 预设: $PRESET"
    log_info "特性配置 -> CPU: 始终启用, Hexagon: $(feature_status $ENABLE_NPU), OpenCL: $(feature_status $ENABLE_GPU), QNN: $(feature_status $ENABLE_QNN), Vulkan: $(feature_status $ENABLE_VULKAN), Profiling: $(feature_status $ENABLE_PROFILING)"
    echo ""

    if [ "$ENABLE_NPU" -eq 1 ] || [ "$ENABLE_QNN_HEXAGON_BACKEND" -eq 1 ]; then
        setup_hexagon_env
    else
        log_info "Hexagon SDK 未使用，跳过配置"
    fi

    if [ "$ENABLE_QNN" -eq 1 ]; then
        setup_qnn_env
    else
        log_info "QNN 被禁用，跳过 QNN SDK 配置"
    fi

    if [ "$ENABLE_VULKAN" -eq 1 ]; then
        setup_vulkan_env
    fi

    configure_build
    build
    install_htp_libs
    verify_build
    show_usage
}

trap 'log_error "编译失败"; exit 1' ERR

main "$@"
