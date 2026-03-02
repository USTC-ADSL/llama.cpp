#!/bin/bash

# ============================================================================
# 脚本功能：将编译好的文件通过ADB传输到Android设备
#
# 功能说明：
# 1. 支持多设备切换
# 2. 传输所有.so库文件
# 3. 传输指定的二进制文件（llama-cli, test-backend-ops）
# 4. 自动添加可执行权限
# ============================================================================

# 注意：不使用 set -e，因为某些命令可能返回非零状态但不是真正的错误

# 配置
BUILD_DIR="build"
BIN_DIR="${BUILD_DIR}/bin"
DEVICE_PATH="/data/local/tmp/llama"

# 需要传输的二进制文件列表
BINARY_FILES=(
    "llama-cli"
    "test-backend-ops"
    "llama-speculative-profile"
    "io-monitor"
)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 函数定义
# ============================================================================

# 打印信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 打印成功
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 打印警告
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 打印错误
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 列出所有连接的设备
list_devices() {
    print_info "连接的设备列表："
    adb devices -l | grep -v "^List" | grep -v "^$"
}

# 选择设备
select_device() {
    local devices=($(adb devices -l | grep -v "^List" | grep -v "^$" | awk '{print $1}'))
    
    if [ ${#devices[@]} -eq 0 ]; then
        print_error "没有找到连接的设备"
        exit 1
    fi
    
    if [ ${#devices[@]} -eq 1 ]; then
        DEVICE="${devices[0]}"
        print_success "自动选择设备: $DEVICE"
        return
    fi
    
    print_info "请选择目标设备:"
    for i in "${!devices[@]}"; do
        echo "  $((i+1)). ${devices[$i]}"
    done
    
    read -p "请输入设备编号 (1-${#devices[@]}): " choice
    
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#devices[@]} ]; then
        print_error "无效的选择"
        exit 1
    fi
    
    DEVICE="${devices[$((choice-1))]}"
    print_success "已选择设备: $DEVICE"
}

# 检查设备连接
check_device() {
    if ! adb -s "$DEVICE" shell echo "test" > /dev/null 2>&1; then
        print_error "无法连接到设备: $DEVICE"
        exit 1
    fi
    print_success "设备连接正常"
}

# 创建设备目录
create_device_dir() {
    print_info "在设备上创建目录: $DEVICE_PATH"
    adb -s "$DEVICE" shell mkdir -p "$DEVICE_PATH"
    print_success "目录创建完成"
}

# 传输.so文件
push_so_files() {
    print_info "开始传输.so库文件..."

    local count=0

    # 使用for循环遍历所有.so文件
    for so_file in "$BIN_DIR"/*.so; do
        # 检查文件是否存在（防止通配符没有匹配到文件的情况）
        if [ ! -f "$so_file" ]; then
            continue
        fi

        local filename=$(basename "$so_file")
        print_info "传输: $filename"
        adb -s "$DEVICE" push "$so_file" "$DEVICE_PATH/$filename"
        ((count++))
    done

    print_success "共传输 $count 个.so文件"
}

# 传输二进制文件
push_binary_files() {
    print_info "开始传输二进制文件..."
    
    local count=0
    for binary in "${BINARY_FILES[@]}"; do
        local binary_path="$BIN_DIR/$binary"
        
        if [ ! -f "$binary_path" ]; then
            print_warning "文件不存在: $binary_path，跳过"
            continue
        fi
        
        print_info "传输: $binary"
        adb -s "$DEVICE" push "$binary_path" "$DEVICE_PATH/$binary"
        ((count++))
    done
    
    print_success "共传输 $count 个二进制文件"
}

# 添加可执行权限
add_executable_permission() {
    print_info "添加可执行权限..."
    
    # 为所有.so文件添加权限
    print_info "为.so文件添加权限..."
    adb -s "$DEVICE" shell "chmod +x $DEVICE_PATH/*.so"
    
    # 为二进制文件添加权限
    print_info "为二进制文件添加权限..."
    for binary in "${BINARY_FILES[@]}"; do
        adb -s "$DEVICE" shell "chmod +x $DEVICE_PATH/$binary"
    done
    
    print_success "权限添加完成"
}

# 验证传输
verify_transfer() {
    print_info "验证传输的文件..."
    
    print_info "设备上的文件列表:"
    adb -s "$DEVICE" shell "ls -lh $DEVICE_PATH"
}

# 显示帮助信息
show_help() {
    cat << EOF
使用方法: $0 [选项]

选项:
    -d, --device <serial>    指定设备序列号（跳过设备选择）
    -l, --list              列出所有连接的设备
    -h, --help              显示此帮助信息

示例:
    $0                      # 交互式选择设备
    $0 -d emulator-5554     # 指定设备
    $0 -l                   # 列出设备

EOF
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    print_info "=========================================="
    print_info "ADB文件传输脚本"
    print_info "=========================================="
    
    # 检查adb是否安装
    if ! command -v adb &> /dev/null; then
        print_error "adb 未安装或不在PATH中"
        exit 1
    fi
    
    # 检查build目录是否存在
    if [ ! -d "$BIN_DIR" ]; then
        print_error "目录不存在: $BIN_DIR"
        exit 1
    fi
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--device)
                DEVICE="$2"
                shift 2
                ;;
            -l|--list)
                list_devices
                exit 0
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 如果未指定设备，则交互式选择
    if [ -z "$DEVICE" ]; then
        list_devices
        echo ""
        select_device
    fi
    
    echo ""
    check_device
    create_device_dir
    push_so_files
    push_binary_files
    add_executable_permission
    verify_transfer
    
    print_success "=========================================="
    print_success "传输完成！"
    print_success "设备路径: $DEVICE_PATH"
    print_success "=========================================="
}

# 运行主程序
main "$@"

