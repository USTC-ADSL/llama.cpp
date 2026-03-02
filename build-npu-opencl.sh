#!/bin/bash

# NPU + OpenCL 编译脚本
# 用于编译支持 Snapdragon NPU 和 OpenCL 的 llama.cpp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="build"
PRESET="arm64-android-snapdragon-release"

# 默认启用 NPU 与 GPU，可通过参数覆盖
ENABLE_NPU=1
ENABLE_GPU=1
ENABLE_PROFILING=0

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

选项:
  --with-npu / --enable-npu        启用 Hexagon NPU (默认)
  --without-npu / --no-npu         禁用 Hexagon NPU
  --with-gpu / --enable-gpu        启用 OpenCL GPU (默认)
  --without-gpu / --no-gpu         禁用 OpenCL GPU
  --with-profiling / --profiling   启用 Stage Profiling (CPU + OpenCL)
  -h, --help                       显示本帮助

示例:
  $0 build-npu arm64-android-snapdragon-release --without-gpu
  $0 build --with-profiling        启用 profiling 进行性能分析
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

# 设置 SDK 环境
setup_sdk_env() {
    log_section "设置 SDK 环境"

    # 设置 Hexagon SDK（使用仓库内的本地 SDK 副本）
    export HEXAGON_SDK_ROOT="$SCRIPT_DIR/hexagon-sdk"
    export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04"

    log_info "HEXAGON_SDK_ROOT: $HEXAGON_SDK_ROOT"
    log_info "HEXAGON_TOOLS_ROOT: $HEXAGON_TOOLS_ROOT"

    # 验证路径
    if [ ! -f "$HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" ]; then
        log_error "Hexagon 工具未找到: $HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang"
        exit 1
    fi
    log_info "✓ Hexagon 工具已验证"
}

# 配置编译
configure_build() {
    log_section "配置编译"

    cd "$SCRIPT_DIR"

    local hexagon_flag="OFF"
    local opencl_flag="OFF"

    if [ "$ENABLE_NPU" -eq 1 ]; then
        hexagon_flag="ON"
    fi

    if [ "$ENABLE_GPU" -eq 1 ]; then
        opencl_flag="ON"
    fi

    log_info "使用预设: $PRESET"
    log_info "构建目录: $BUILD_DIR"
    log_info "Hexagon NPU: $(feature_status $ENABLE_NPU)"
    log_info "OpenCL GPU: $(feature_status $ENABLE_GPU)"
    log_info "Stage Profiling: $(feature_status $ENABLE_PROFILING)"

    rm -rf "$BUILD_DIR"

    local cmake_args=(
        --preset "$PRESET"
        -B "$BUILD_DIR"
        -DGGML_HEXAGON="$hexagon_flag"
        -DGGML_OPENCL="$opencl_flag"
    )

    # 添加 profiling 选项
    if [ "$ENABLE_PROFILING" -eq 1 ]; then
        cmake_args+=(
            -DGGML_OPENCL_PROFILING=ON
            -DGGML_CPU_PROFILING=ON
        )
    fi

    if [ "$ENABLE_NPU" -eq 1 ]; then
        cmake_args+=(
            -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT"
            -DHEXAGON_TOOLS_ROOT="$HEXAGON_TOOLS_ROOT"
        )
    fi

    cmake "${cmake_args[@]}"

    log_info "✓ 配置完成"
}

# 构建
build() {
    log_section "构建"

    cd "$SCRIPT_DIR"

    local num_jobs=$(nproc 2>/dev/null || echo 4)
    log_info "使用 $num_jobs 个并行任务"

    cmake --build "$BUILD_DIR" -j "$num_jobs"

    log_info "✓ 构建完成"
}

# 安装 HTP 库
install_htp_libs() {
    log_section "安装 HTP 库"

    if [ "$ENABLE_NPU" -ne 1 ]; then
        log_info "已禁用 NPU，跳过 HTP 库复制"
        return
    fi

    local hexagon_build_dir="$SCRIPT_DIR/$BUILD_DIR/ggml/src/ggml-hexagon"
    local bin_dir="$SCRIPT_DIR/$BUILD_DIR/bin"

    # 复制 HTP 库到 bin 目录
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

# 验证输出
verify_build() {
    log_section "验证输出"

    local exe="$SCRIPT_DIR/$BUILD_DIR/bin/llama-cli"

    if [ -f "$exe" ]; then
        log_info "✓ 可执行文件已生成: $exe"
        log_info "文件大小: $(du -h "$exe" | cut -f1)"

        # 检查后端库
        log_info "后端库:"
        if [ "$ENABLE_NPU" -eq 1 ]; then
            local hex_lib="libggml-hexagon.so"
            if [ -f "$SCRIPT_DIR/$BUILD_DIR/bin/$hex_lib" ]; then
                log_info "  ✓ $hex_lib ($(du -h "$SCRIPT_DIR/$BUILD_DIR/bin/$hex_lib" | cut -f1))"
            else
                log_warn "  ✗ $hex_lib 缺失"
            fi
        else
            log_info "  - 已禁用 Hexagon NPU"
        fi

        if [ "$ENABLE_GPU" -eq 1 ]; then
            local ocl_lib="libggml-opencl.so"
            if [ -f "$SCRIPT_DIR/$BUILD_DIR/bin/$ocl_lib" ]; then
                log_info "  ✓ $ocl_lib ($(du -h "$SCRIPT_DIR/$BUILD_DIR/bin/$ocl_lib" | cut -f1))"
            else
                log_warn "  ✗ $ocl_lib 缺失"
            fi
        else
            log_info "  - 已禁用 OpenCL GPU"
        fi

        local cpu_lib="libggml-cpu.so"
        if [ -f "$SCRIPT_DIR/$BUILD_DIR/bin/$cpu_lib" ]; then
            log_info "  ✓ $cpu_lib ($(du -h "$SCRIPT_DIR/$BUILD_DIR/bin/$cpu_lib" | cut -f1))"
        fi

        # 检查 HTP 库
        if [ "$ENABLE_NPU" -eq 1 ]; then
            log_info "HTP 库 (Hexagon DSP 运行时):"
            for version in v73 v75 v79 v81; do
                local htp_lib="libggml-htp-${version}.so"
                if [ -f "$SCRIPT_DIR/$BUILD_DIR/bin/$htp_lib" ]; then
                    log_info "  ✓ $htp_lib ($(du -h "$SCRIPT_DIR/$BUILD_DIR/bin/$htp_lib" | cut -f1))"
                else
                    log_warn "  ✗ $htp_lib"
                fi
            done
        else
            log_info "HTP 库: 已禁用 NPU"
        fi

        # 检查符号
        if command -v readelf &> /dev/null; then
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

            readelf -d "$exe" 2>/dev/null | grep -q "libggml-cpu.so" && log_info "  ✓ CPU 后端" || log_warn "  ✗ CPU 后端"
        fi
    else
        log_error "可执行文件未生成: $exe"
        exit 1
    fi
}

# 显示使用信息
show_usage() {
    log_section "编译完成"

    local exe="$SCRIPT_DIR/$BUILD_DIR/bin/llama-cli"

    echo "构建参数:"
    echo "  NPU: $(feature_status $ENABLE_NPU)"
    echo "  GPU: $(feature_status $ENABLE_GPU)"
    echo "  Profiling: $(feature_status $ENABLE_PROFILING)"
    echo "  目录: $BUILD_DIR"
    echo ""

    echo "使用示例:"
    echo ""
    echo "  # 基本推理"
    echo "  $exe -m model.gguf -p \"Hello world\""
    echo ""
    echo "  # 使用 OpenCL 加速"
    echo "  $exe -m model.gguf -p \"Hello world\" -ngl 33"
    echo ""
    echo "  # 使用 Hexagon NPU"
    echo "  $exe -m model.gguf -p \"Hello world\" --hexagon"
    echo ""
    echo "  # 交互模式"
    echo "  $exe -m model.gguf -i"
    echo ""
}

# 主函数
main() {
    parse_args "$@"

    log_info "llama.cpp NPU + OpenCL 编译脚本"
    log_info "构建目录: $BUILD_DIR | 预设: $PRESET"
    log_info "特性配置 -> NPU: $(feature_status $ENABLE_NPU), GPU: $(feature_status $ENABLE_GPU), Profiling: $(feature_status $ENABLE_PROFILING)"
    echo ""

    if [ "$ENABLE_NPU" -eq 1 ]; then
        setup_sdk_env
    else
        log_info "NPU 被禁用，跳过 Hexagon SDK 配置"
    fi

    configure_build
    build
    install_htp_libs
    verify_build
    show_usage
}

# 错误处理
trap 'log_error "编译失败"; exit 1' ERR

main "$@"
