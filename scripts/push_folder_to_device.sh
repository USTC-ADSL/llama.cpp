#!/bin/bash

# =============================================================================
# push_folder_to_device.sh
#
# 更灵活的 ADB 同步脚本，可通过交互方式选择本地目录、目标目录以及设备。
# - 默认同步 build-new/bin 下的内容到 /data/local/tmp/llama
# - 交互提示允许输入绝对/相对路径，并打印当前工作目录
# - 支持推送目录或单文件，并可交互式选择“镜像”和“清理”策略
# =============================================================================

# 不使用 set -e，方便容错

# 默认参数
SOURCE_DIR="build-new/bin"
TARGET_DIR="/data/local/tmp/llama"
DEVICE=""
MIRROR_MODE=false
CLEAN_TARGET=false
SOURCE_IS_DIR=true

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }

check_adb() {
    if ! command -v adb >/dev/null 2>&1; then
        print_error "adb 未安装或不在 PATH 中"
        exit 1
    fi
}

list_devices() {
    print_info "当前连接的设备:" 
    adb devices -l | grep -v "^List" | grep -v "^$" || print_warn "未检测到设备"
}

select_device() {
    local devices=($(adb devices -l | grep -v "^List" | grep -v "^$" | awk '{print $1}'))

    if [ ${#devices[@]} -eq 0 ]; then
        print_error "没有可用设备"
        exit 1
    fi

    if [ ${#devices[@]} -eq 1 ]; then
        DEVICE="${devices[0]}"
        print_ok "自动选择设备: $DEVICE"
        return
    fi

    print_info "请选择目标设备:"
    local idx=1
    for serial in "${devices[@]}"; do
        echo "  $idx. $serial"
        idx=$((idx+1))
    done

    read -p "请输入编号 (1-${#devices[@]}): " choice
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#devices[@]} ]; then
        print_error "无效的选择"
        exit 1
    fi

    DEVICE="${devices[$((choice-1))]}"
    print_ok "已选择设备: $DEVICE"
}

check_device_connection() {
    if ! adb -s "$DEVICE" shell echo ping >/dev/null 2>&1; then
        print_error "无法连接到设备 $DEVICE"
        exit 1
    fi
    print_ok "设备连接正常"
}

prompt_local_dir() {
    local cwd
    cwd=$(pwd)
    print_info "当前工作目录: $cwd"
    read -p "请输入要推送的本地目录 (默认: ${SOURCE_DIR}): " input_dir
    if [ -z "$input_dir" ]; then
        input_dir="$SOURCE_DIR"
    fi

    if [[ "$input_dir" != /* ]]; then
        input_dir="${cwd}/${input_dir}"
    fi

    local real_path
    real_path=$(realpath -m "$input_dir" 2>/dev/null)

    if [ -d "$real_path" ]; then
        SOURCE_IS_DIR=true
    elif [ -f "$real_path" ]; then
        SOURCE_IS_DIR=false
    else
        print_error "路径不存在: $real_path"
        exit 1
    fi

    SOURCE_DIR="$real_path"
    if $SOURCE_IS_DIR; then
        print_ok "选择本地目录: $SOURCE_DIR"
    else
        print_ok "选择本地文件: $SOURCE_DIR"
    fi
}

prompt_target_dir() {
    read -p "请输入设备上的目标绝对路径 (默认: ${TARGET_DIR}): " target_input
    if [ -z "$target_input" ]; then
        target_input="$TARGET_DIR"
    fi

    if [[ "$target_input" != /* ]]; then
        print_error "目标路径必须为绝对路径"
        exit 1
    fi

    TARGET_DIR="$target_input"
    print_ok "目标目录: $TARGET_DIR"
}

prompt_toggle_options() {
    read -p "使用镜像推送模式? (y/N): " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        MIRROR_MODE=true
    else
        MIRROR_MODE=false
    fi

    read -p "推送前清理目标目录? (y/N): " clean_answer
    if [[ "$clean_answer" =~ ^[Yy]$ ]]; then
        CLEAN_TARGET=true
    else
        CLEAN_TARGET=false
    fi
}

prepare_target_dir() {
    print_info "在设备上创建目录: $TARGET_DIR"
    adb -s "$DEVICE" shell "mkdir -p '$TARGET_DIR'"

    if $CLEAN_TARGET; then
        print_info "清理目标目录内容"
        adb -s "$DEVICE" shell "rm -rf '$TARGET_DIR'/*"
    fi
}

push_directory_contents() {
    if $SOURCE_IS_DIR; then
        if [ ! -d "$SOURCE_DIR" ]; then
            print_error "本地目录不存在: $SOURCE_DIR"
            exit 1
        fi

        if $MIRROR_MODE; then
            print_info "镜像推送目录: $SOURCE_DIR -> $TARGET_DIR"
            adb -s "$DEVICE" push "$SOURCE_DIR" "$TARGET_DIR"
        else
            print_info "推送目录内容: $SOURCE_DIR/. -> $TARGET_DIR"
            adb -s "$DEVICE" push "$SOURCE_DIR"/. "$TARGET_DIR"
        fi
    else
        if $MIRROR_MODE; then
            print_warn "单文件推送不支持镜像模式，将直接复制该文件。"
        fi
        if [ ! -f "$SOURCE_DIR" ]; then
            print_error "文件不存在: $SOURCE_DIR"
            exit 1
        fi

        local dest="$TARGET_DIR"
        case "$dest" in
            */) ;;
            *) dest="${dest}/" ;;
        esac

        print_info "推送文件: $SOURCE_DIR -> $dest"
        adb -s "$DEVICE" push "$SOURCE_DIR" "$dest"
    fi
}

verify_remote() {
    print_info "目标目录文件列表:"
    adb -s "$DEVICE" shell "ls -lh '$TARGET_DIR'"
}

main() {
    check_adb

    list_devices

    if [ -z "$DEVICE" ]; then
        select_device
    fi

    check_device_connection
    prompt_local_dir
    prompt_target_dir
    prompt_toggle_options
    prepare_target_dir
    push_directory_contents
    verify_remote

    print_ok "传输完成 (设备: $DEVICE)"
}

main "$@"
