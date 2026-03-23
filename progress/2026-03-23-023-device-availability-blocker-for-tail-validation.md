# 2026-03-23-023 真机 tail validation 设备可用性阻塞

## 任务

在用户说明“设备已上线”后，继续对最后一层 `residual / fragment` stopgap 做真机复测。

目标仍然是验证：

- 路线：`attn_proj=cpu,attn_core=qnn-npu,attn_out=qnn-npu,ffn=cpu,output=cpu`
- 模型：优先 `Qwen2`
- 关注点：
  - `unmatched cgraph` 是否消失
  - residual `OpenCL split` 是否保持为 0
  - mixed route 是否出现明显吞吐回退

## 执行内容

### 1. 按规范重新构建

执行：

- `./build-npu-opencl.sh build-qnn-current-verify arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn`

结果：

- 构建成功
- `build-qnn-current-verify/bin/llama-bench`
- `build-qnn-current-verify/bin/libllama.so`
- `build-qnn-current-verify/bin/libggml-qnn.so`

均已重新生成。

### 2. 检查设备可见性

执行：

- `adb -s db6c02cf get-state`
- `adb -s fd8657d6 get-state`
- `adb devices -l`
- `adb connect 192.168.50.85:5555`

结果：

- `adb -s db6c02cf get-state`：超时
- `adb -s fd8657d6 get-state`：超时
- `adb devices -l`：空列表
- `adb connect 192.168.50.85:5555`：超时

说明当前 host 侧并没有看到可用设备。

## 结论

当前真机测试**未能开始**，原因不是代码或构建失败，而是：

- ADB 没有枚举到任何在线设备

按仓库约束：

- “若设备不在线，自动停止工作告知用户来处理”

因此这一轮必须停在设备可用性这里，不能继续假设性地执行推送和 benchmark。

## 下一步

设备恢复后，直接继续下面这条最小复测链即可：

1. `adb devices -l` 确认能看到设备
2. 推送 `build-qnn-current-verify/bin/llama-bench` 与相关 `.so`
3. 运行 `Qwen2` 的 `cpu/qnn/cpu` 复测口径
4. 检查日志中是否还存在：
   - `unmatched cgraph: n_nodes=18 first=cache_k_upd-23 last=attn_out-tail-23`
   - `unmatched cgraph: n_nodes=1 first=ffn_inp-23 last=ffn_inp-23`

若你确认设备现在已经能在本机 `adb devices -l` 中看到，我下一步就直接继续推送和跑测，不需要再做额外准备。
