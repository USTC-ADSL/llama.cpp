# 2026-03-23-027 Qwen3 `ffn=qnn-npu` Bootstrap Scheduler Fix

## 任务

修复 `Qwen3` 在 mixed route

- `attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=cpu`

下的 `ffn=qnn-npu` 失败问题，并确认它不是旧的 AoT 路径解析问题复发，而是 bootstrap correction 与预分配权重之间的新调度冲突。

这项工作服务于当前 decode-first 主线：

- 问题发生在 decode `tg1` 路径；
- 关注的阶段边界是 `attn(OpenCL) -> ffn(QNN) -> output(CPU)`；
- 核心不是“理论上 FFN 能不能上 NPU”，而是 mixed route 的 runtime overhead / scheduler correctness 是否允许该阶段级路由实际落地。

## 根因

前一轮已经修复了 AoT combined config 的相对路径解析问题，`ffn` 子图本身已经能被匹配并真实执行到 QNN。

剩余失败点出现在 initial decode token 的 `AoT bootstrap CPU correction`：

1. 运行时会先用 steady-state mixed graph 跑一遍首 token。
2. 然后为了做 correction，`process_ubatch()` 会把 `sched` 临时替换成一个 CPU-only scheduler。
3. 但在 `-ngl 99 -dev GPUOpenCL` 下，模型权重已经预分配到了 `OpenCL` buffer。
4. CPU-only scheduler 在重新分图时看见：
   - `blk.0.attn_q.weight` 已经在 `OpenCL` buffer 中；
   - 自己却只剩 CPU backend；
   - 因而直接在 `ggml/src/ggml-backend.cpp:992` abort：
     - `pre-allocated tensor (blk.0.attn_q.weight) in a buffer (OpenCL) that cannot run the operation (NONE)`

所以这里的失败不是：

- `Qwen3` FFN matcher 错了；
- `ffn=qnn-npu` 悄悄退回 CPU 了；
- 或者 OpenCL/QNN 之间又出现了张量显式 memcpy。

更准确地说，这是 bootstrap correction 的 scheduler 设计默认假设“模型权重仍在 CPU”，而这个假设在 `-ngl > 0` 的 mixed route 下不成立。

## 代码修改

### 1. correction scheduler 改成按权重驻留状态分支

文件：

- [src/llama-context.cpp](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/src/llama-context.cpp)

修改后：

- 如果 `model.n_gpu_layers() == 0`
  - 仍保持原来的 CPU-only bootstrap correction 路径；
  - 这保证此前已经验证过的 decode purity 结论不被回退。
- 如果 `model.n_gpu_layers() > 0`
  - 不再把 `sched` 替换成 CPU-only scheduler；
  - 保留 steady-state scheduler；
  - 打日志：
    - `bootstrap correction keeps the steady-state scheduler because n_gpu_layers=... leaves model weights pre-allocated on non-CPU backends`

### 2. correction 期间只把 QNN 负责的阶段压回 CPU

同文件：

- [src/llama-context.cpp](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/src/llama-context.cpp)

修改后：

- `aot_force_cpu_graph` 不再无条件把所有张量都标成 CPU；
- 只对 correction 里真正的阶段张量生效；
- 如果该阶段原本是 QNN backend，则标记为：
  - `reason=bootstrap-qnn-cpu`
- 如果该阶段原本已经是非 QNN 的 offloaded backend，例如 `OpenCL` attention，则保留原 steady-state backend，不去碰它的预分配权重链路。

这一步避免了另一个潜在错误：

- 如果 correction 继续粗暴地把叶子权重张量也标成 CPU，就只是把 abort 从“CPU-only scheduler 看不见 OpenCL weight”换成“预分配 OpenCL weight 被错误强绑到 CPU”。

### 3. 说明注释同步更新

文件：

- [src/llama-context.h](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/src/llama-context.h)

把 `aot_force_cpu_graph` 的注释改成了更准确的语义：

- CPU-only 只是在“所有权重仍驻留 CPU”时成立；
- 否则 correction 会保留已有 offloaded backend，只把 QNN-owned stages 重新路由。

## 构建

按规范重新执行了：

- `./build-npu-opencl.sh build-qnn-current-verify arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn --with-profiling`

关键新产物时间戳：

- `build-qnn-current-verify/bin/libllama.so`：`2026-03-23 12:55:59 UTC`
- `build-qnn-current-verify/bin/llama-bench`：`2026-03-23 12:56:17 UTC`

然后同步到设备：

- 设备：`fd8657d6`
- 目录：`/data/local/tmp/acom-fd-qwen23-verify/bin/`

## 设备验证

### 设备状态

- `adb devices` 显示 `fd8657d6` 在线。
- 但 `dumpsys power` 显示：
  - `mWakefulness=Asleep`
- `shell` 用户没有 `INJECT_EVENTS` 权限；
- 设备上也没有可用的 `su`；
- 因此这轮无法从 ADB 侧强制亮屏。

所以本轮设备结果应视为：

- 功能性 / correctness 验证有效；
- 严格性能口径需要在亮屏满足后复测。

### A. mixed `OpenCL + ffn=qnn-npu` 复测通过

命令口径：

- 模型：`/data/local/tmp/Qwen3-1.7B-Q8_0.gguf`
- 路线：
  - `attn_proj=opencl,attn_core=opencl,attn_out=opencl,ffn=qnn-npu,output=cpu`
- AoT config：
  - `/data/local/tmp/acom-fd-qwen23-verify/models/qwen3_1.7b/models/Qwen3-AoT/qwen3-qnn-full/qnn_ffn_combined.json`
- bench：
  - `-p 0 -n 1 -b 1 -ub 1 -t 1 -r 1`
  - `-ngl 99 -dev GPUOpenCL --mmap 0`
  - `-ctk f32 -ctv f32 -c 2048`

结果日志：

- `tmp/qwen3_ffn_fix_20260323/qwen3_openclattn_ffnqnn_tg1_after_sched_fix.log`

关键证据：

- `process_ubatch: running AoT bootstrap CPU correction for initial decode token`
- `process_ubatch: bootstrap correction keeps the steady-state scheduler because n_gpu_layers=99 leaves model weights pre-allocated on non-CPU backends`
- 所有层都继续出现：
  - `execute ffn graph ffn_layer_X_batch_1`
  - `direct_bind_result x=1 out=1`
- 日志中不再出现：
  - `pre-allocated tensor (blk.0.attn_q.weight) in a buffer (OpenCL) that cannot run the operation (NONE)`

bench 结果：

- `tg1 = 13.21 ± 0.00 tok/s`

### B. 原 `-ngl 0` 纯 FFN 路线未回退

命令口径：

- 路线：
  - `attn_proj=cpu,attn_core=cpu,attn_out=cpu,ffn=qnn-npu,output=cpu`
- bench：
  - `-ngl 0 --mmap 0`
  - 其它参数保持同一 `tg1` smoke 口径

结果日志：

- `tmp/qwen3_ffn_fix_20260323/qwen3_cpuattn_ffnqnn_tg1_ngl0_after_sched_fix.log`

关键证据：

- 仍有所有层的：
  - `execute ffn graph ffn_layer_X_batch_1`
  - `direct_bind_result x=1 out=1`
- bootstrap correction 结束后仍能看到：
  - `synchronize: restoring steady-state scheduler after AoT bootstrap CPU correction`

bench 结果：

- `tg1 = 6.05 ± 0.00 tok/s`

这说明新补丁没有破坏原本已经修好的 `-ngl 0` FFN AoT 路线。

## 判断

本轮已经形成一个强结论：

- `Qwen3` 的 `ffn=qnn-npu` 原始失败原因可以分成两层：
  1. 先前的 AoT config 相对路径解析错误；
  2. 修掉路径后，`-ngl 99` mixed route 还会被 bootstrap CPU-only scheduler 与 OpenCL 预分配权重的冲突再次打崩。

目前两层都已经闭环：

- `ffn` AoT 图能够被正确加载；
- mixed `OpenCL + QNN FFN` 路线也能完整跑通 decode `tg1`；
- 且没有通过把 `ffn=qnn-npu` 偷偷降回 CPU/OpenCL 来“伪修复”。

## 对主线的意义

这项修复直接支持当前故事主线中的第 4 点：

- runtime overhead / scheduler consistency 本身就是释放阶段级异构收益的关键瓶颈。

这里的经验不是“FFN 一上 NPU 就一定更优”，而是：

- 即便阶段级 backend 选择本身是合理的；
- 只要 bootstrap / scheduler 的假设仍按“全 CPU 权重”写死；
- mixed route 仍然会在 runtime 上直接失败，连收益讨论都无法成立。

## 仍缺的数据

要把这件事升级成更强的系统级结论，还差两类数据：

1. 亮屏条件满足后的 clean 性能复测
   - 当前 `tg1` 数值不应作为严格性能口径；
   - 需要在亮屏状态下重新测 mixed / static baseline。
2. 更长 decode 长度下的 mixed route 性能与 overhead 归因
   - 目前只完成了 `tg1` smoke；
   - 还需要看 `tg64` 等更接近主线的 decode 区间，确认：
     - FFN 上 QNN 的潜在收益是否会被同步 / split / CPU output tail 抵消。
