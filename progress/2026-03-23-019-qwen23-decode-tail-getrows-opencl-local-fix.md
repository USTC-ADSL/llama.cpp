# 2026-03-23-019 Qwen2/Qwen3 Decode Tail `GET_ROWS` OpenCL Local Fix

## 任务

简单判断当前 decode 路径里为什么还会出现 `OpenCL` 参与，并在不改动大块 scheduler 逻辑的前提下，先做一版最小修复尝试。

## 结论

- 这次看到的 decode `OpenCL` residual split，当前证据更支持它来自最后一层输出裁剪路径上的匿名 `GET_ROWS`，而不是 `qwen3` 与 `qwen2` 的 `attn-core` 计算图差异本身。
- 具体路径是最后一层 `attn_out` / 上一层 `l_out` 在 `ffn_inp` 之前的 `ggml_get_rows()`。
- 由于这两个节点没有经过 `cb()` 命名，现有 stage route / purity fallback 逻辑抓不到它们；在 `GGML_HETERO_QNN_SHARED_HOST=1` 下，scheduler 会把这两个未钉住的小节点抬到 `OpenCL`。

## 完成内容

- 在 [qwen2.cpp](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/src/models/qwen2.cpp) 中，为最后一层输出裁剪后的两个 `GET_ROWS` 补了显式命名：
  - `attn_out-tail`
  - `l_out-tail`
- 在 [qwen3.cpp](/mnt/sda1/pzw/HeteroCompute/llama.cpp-acom/src/models/qwen3.cpp) 中：
  - 补上了缺失的 `attn_out` 命名；
  - 为最后一层输出裁剪后的两个 `GET_ROWS` 同样补了 `attn_out-tail` / `l_out-tail` 命名。
- 已按当前实验规范重新执行：
  - `./build-npu-opencl.sh build-qnn-prof-db arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn --with-profiling`

## 预期收益

- 让 `qwen2/qwen3` 最后一层输出裁剪节点进入现有的 stage routing 逻辑。
- 当 `attn_out=qnn-npu` 但 `GET_ROWS` 不被 QNN 支持、且当前 route 并未显式使用 `OpenCL` 时，这两个 tail 节点应走现有的 CPU fallback，而不再被 scheduler 自动抬到 `OpenCL`。
- 这项修复服务于 decode 主线里的 route purity 和 runtime overhead 解释收口，不直接宣称端到端性能已经改善。

## 风险与边界

- 这次只是局部 builder 标注修复，没有改 scheduler 优先级，也没有改 cost model。
- 即使 `OpenCL` residual 消失，`attn_out -> CPU tail GET_ROWS -> qnn/CPU 后续阶段` 的跨后端同步与搬移成本仍然可能存在。
- 因此它更像是“收口一个不该出现的 `OpenCL` 小 split”，而不是已经证明 decode 路线端到端更优。

## 仍缺的数据

- 还需要真机按原 purity 口径复测，确认：
  - `OpenCL` 小 split 是否消失；
  - `qwen2` 与 `qwen3` 是否都收口；
  - 是否出现新的 CPU/QNN 小 split 或 unmatched graph。
- 当前这条记录的强结论只到“局部根因已定位，并完成最小代码修复尝试 + 编译通过”。
