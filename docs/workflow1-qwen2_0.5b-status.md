# Workflow1 qwen2_0.5B AoT 当前状态（2026-03-17）

## 目标
先把 `workflow1` 在 `qwen2_0.5B` 上稳定下来，再讨论性能和后续 `workflow2` 的共享内存优化。

## 当前复现配置
- 设备：`192.168.50.85:5555`
- 模型：`/data/local/tmp/llama-qnn-aot/models/qwen2_0.5b/ggml/weights.gguf`
- AoT 配置：`/data/local/tmp/llama-qnn-aot/models/qwen2_0.5b/qnn/config.json`
- 典型命令：
  - 构建：`./build-npu-opencl.sh build-qnn-wf1-verify arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn`
  - 运行：
    - `taskset 80 ./llama-cli -m /data/local/tmp/llama-qnn-aot/models/qwen2_0.5b/ggml/weights.gguf -ngl 99 -dev qnn-npu -t 1 -c 256 -p "hi" -n 4 --no-mmap`
- 关键环境变量：
  - `GGML_QNN_AOT_CONFIG`
  - `GGML_QNN_AOT_MODEL_DIR`
  - `LD_LIBRARY_PATH`
  - `ADSP_LIBRARY_PATH`

## 当前结论
1. **固定 shape AoT 是约束，但不是当前 decode 不稳定的主因。**
   - PowerServe 的 `qwen2_0.5B` AoT 配置里同时存在 `batch_1` 和 `batch_128`。
   - 当前本地 AoT runtime 也已经有：
     - `1 token -> batch_1`
     - `2..128 token -> batch_128`
     - `>128 token -> 分块执行`
     - 小于 graph batch 的场景会先清零再只填充前 `step` 行，等价于 padding。

2. **当前主问题更像是 scheduler 把 decode 阶段切碎了。**
   - 之前残余图表现为：`ROPE(Qcur-0)` fallback。
   - 随着修改推进，残余未匹配图继续变化：
     - `n_nodes=23 first=norm-0 last=cache_v_l0 (view) (permuted)`
     - `n_nodes=14 first=norm-0 last=Kcur-0 (view)`
     - 当前最新：`n_nodes=2 first=norm-0 last=attn_norm-0`
   - 这说明问题已经不只是某一个算子不支持，而是 **AoT 期望的完整 transformer 子图没有被保留下来**。

3. **当前最危险的不是“fallback 到 CPU”，而是“fallback 到 QNN JIT 小图”。**
   - 目前未匹配残余图会进入 QNN JIT 路径。
   - 实际观察到 tiny graph 在 QNN 侧失败：
     - `QNN_GRAPH_ERROR_MEM_ALLOC`
     - `QNN_GRAPH_ERROR_INVALID_NAME`
   - 因此当前首要目标是稳定性：要么保证整段 AoT 命中，要么未命中时明确走 CPU，而不是继续把碎图交给 QNN JIT。

## 对“固定 128 / padding / 切分”问题的判断
- **不是主要根因。**
- PowerServe 的固定 batch 策略本身是可行的，前提是 runtime 看到的仍然是完整且可识别的阶段输入输出。
- 当前更像是 ggml scheduler 先把 decode 图按后端/算子能力切成碎片，AoT runtime 拿到的已经不是一个完整的 transformer-stage cgraph。

## 对“残余图直接放 CPU 上算是否影响不大”的判断
- **可以作为稳定性止血手段，但不能认为“影响不会很多”。**
- 原因：
  1. 这些残余图位于 decode 关键路径，Attention/FFN/KV 都在每 token 重复执行。
  2. 一旦残余图落到 CPU，会引入额外的后端切换、同步和张量搬移。
  3. 如果 AoT 命中的阶段被切得过碎，理论上的 NPU 优势很容易被 runtime overhead 抵消。
  4. 当前更糟的是它们还不总是稳定落到 CPU，而是可能继续走到 QNN JIT 小图并失败。
- 因此，**CPU fallback 可以是稳定性兜底，但不能当成对性能“基本无影响”的默认假设。**

## 当前缺口
- 还缺一条稳定策略，保证 decode 阶段：
  - 要么整段由 AoT 执行；
  - 要么残余图统一落 CPU；
  - 避免碎片图落入 QNN JIT。
- 还缺端到端的 `llama-bench` 数据来量化：
  - CPU fallback 残余比例对 decode tok/s 的影响；
  - 是否还能保持 `qwen2_0.5B` 的稳定 decode。

## 下一步最小可验证方案
1. 先继续围绕 `workflow1` 做稳定性修复，不切到 `workflow2`。
2. 优先把 **AoT 未匹配残余图统一导向 CPU fallback**，避免 QNN JIT 小图失败。
3. 再观察残余图的形态是否继续缩小，判断是否需要进一步上移 AoT 的阶段边界，绕开 scheduler 的细粒度切分。
4. 稳定后再用 `llama-bench` 测 `qwen2_0.5B` 的 decode/prefill，判断性能损失主要来自：
   - CPU fallback 频率
   - backend split/sync
   - KV cache 读写与数据搬移
