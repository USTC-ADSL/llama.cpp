#!/bin/bash

# 简化版本的ADB文件传输脚本
# 用法: ./push_to_device_simple.sh [设备序列号]

set -e

BUILD_DIR="build-new/bin"
DEVICE_PATH="/data/local/tmp/llama"
BINARY_FILES=("llama-cli" "test-backend-ops")

# 获取设备
if [ -z "$1" ]; then
    echo "可用设备:"
    adb devices -l | grep -v "^List" | grep -v "^$"
    echo ""
    read -p "输入设备序列号: " DEVICE
else
    DEVICE="$1"
fi

echo "使用设备: $DEVICE"

# 创建目录
echo "创建设备目录..."
adb -s "$DEVICE" shell mkdir -p "$DEVICE_PATH"

# 传输.so文件
echo "传输.so库文件..."
for so_file in "$BUILD_DIR"/*.so; do
    if [ -f "$so_file" ]; then
        echo "  -> $(basename "$so_file")"
        adb -s "$DEVICE" push "$so_file" "$DEVICE_PATH/"
    fi
done

# 传输二进制文件
echo "传输二进制文件..."
for binary in "${BINARY_FILES[@]}"; do
    if [ -f "$BUILD_DIR/$binary" ]; then
        echo "  -> $binary"
        adb -s "$DEVICE" push "$BUILD_DIR/$binary" "$DEVICE_PATH/"
    fi
done

# 添加权限
echo "添加可执行权限..."
adb -s "$DEVICE" shell "chmod +x $DEVICE_PATH/*.so $DEVICE_PATH/llama-cli $DEVICE_PATH/test-backend-ops"

# 验证
echo ""
echo "设备上的文件:"
adb -s "$DEVICE" shell "ls -lh $DEVICE_PATH"

echo ""
echo "✓ 传输完成！"

