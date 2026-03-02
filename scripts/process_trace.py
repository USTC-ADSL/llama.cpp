import json
import argparse
import os
import sys

def load_and_clean_trace(file_path):
    """
    读取json文件，具备自动修复格式错误功能。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # === 自动修复 JSON 格式错误 ===
        if content.endswith(','):
            content = content[:-1]
        if not content.endswith(']'):
            last_bracket_index = content.rfind('}')
            if last_bracket_index != -1:
                content = content[:last_bracket_index+1] + ']'
            else:
                content += ']'
        if not content.startswith('['):
            if not content.startswith('{'):
                content = '[' + content

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print("标准解析失败，尝试流式逐行解析...")
            data = []
            raw_lines = content.split('\n')
            for line in raw_lines:
                line = line.strip()
                if line == '[' or line == ']': continue
                if line.endswith(','): line = line[:-1]
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except:
                    continue
        
        if isinstance(data, dict) and 'traceEvents' in data:
            data = data['traceEvents']
            
    except Exception as e:
        print(f"读取文件发生无法处理的错误: {e}")
        exit(1)

    # 1. 过滤 Device 线程
    if not isinstance(data, list):
        print("错误：JSON解析后不是列表格式，无法继续。")
        exit(1)

    device_events = [e for e in data if e.get('tid') == 'Device' and e.get('cat') == 'OpenCL']
    device_events.sort(key=lambda x: x['ts'])

    # 2. 合并 B 和 E 为完整事件
    processed_kernels = []
    stack = []

    for event in device_events:
        if event['ph'] == 'B':
            stack.append(event)
        elif event['ph'] == 'E':
            if stack:
                start_event = stack.pop()
                if start_event['name'] != event['name']:
                    continue
                duration = event['ts'] - start_event['ts']
                processed_kernels.append({
                    'name': start_event['name'],
                    'ts': start_event['ts'],
                    'dur': duration
                })
    
    processed_kernels.sort(key=lambda x: x['ts'])
    return processed_kernels

def analyze_blocks(kernels):
    """
    基于语义（FlashAttn -> SwiGLU -> Add）来划分 Block。
    不再依赖 Add 的计数，而是依赖执行顺序。
    """
    blocks = []
    current_block_kernels = []
    layer_index = 0
    
    # 状态标志，用于判断当前处于 Block 的哪个部分
    has_seen_flash = False
    has_seen_swiglu = False
    
    # Layer 28 (index 27) 尾部处理
    capturing_layer_28_tail = False
    layer_28_tail_count = 0 
    
    for k in kernels:
        name = k['name']
        current_block_kernels.append(k)
        
        # === 1. 特殊处理 Layer 27 的尾部 (最后两个算子) ===
        if capturing_layer_28_tail:
            layer_28_tail_count += 1
            # Layer 27 在最后的 Add 之后还有 rms_norm 和 mul_mv
            if layer_28_tail_count >= 2:
                blocks.append({
                    'layer': layer_index,
                    'kernels': current_block_kernels
                })
                # 重置所有状态，准备下一个 Token (Layer 0)
                current_block_kernels = []
                has_seen_flash = False
                has_seen_swiglu = False
                capturing_layer_28_tail = False
                layer_28_tail_count = 0
                layer_index = 0 
            continue

        # === 2. 状态更新 ===
        if "flash_attn" in name:
            has_seen_flash = True
        
        # 检测 FFN 核心计算
        # 注意：Prefill用 mul_mm, Decode用 mul_mv, 这里用 swiglu 作为最强特征
        # 如果 swiglu 被融合了，可以用 "出现在 flash_attn 之后的大型 mul" 来辅助，但目前假设 swiglu 存在
        if "swiglu" in name:
            has_seen_swiglu = True
            
        # === 3. 检测 Block 结束边界 ===
        # 只有当我们已经经过了 FlashAttn 和 FFN(SwiGLU) 阶段，遇到的 Add 才是 Block 的结尾
        if name.startswith("kernel_add"):
            if has_seen_flash and has_seen_swiglu:
                # 这是一个 Block 的结束 Add (FFN Residual)
                
                if layer_index == 27:
                    # 如果是最后一层，不要立即切分，进入捕获尾部模式
                    capturing_layer_28_tail = True
                else:
                    # 普通层，直接切分
                    blocks.append({
                        'layer': layer_index,
                        'kernels': current_block_kernels
                    })
                    # 重置状态，准备下一层
                    current_block_kernels = []
                    has_seen_flash = False
                    has_seen_swiglu = False
                    layer_index += 1

    return blocks

def process_single_block(block_data):
    """
    计算四个阶段的时间。
    阶段划分逻辑更新：
    P1: Start -> rope (exclusive)
    P2: rope (inclusive) -> flash_attn (exclusive)
    P3: flash_attn (inclusive) -> [FFN Start RMS_NORM] (exclusive)
    P4: [FFN Start RMS_NORM] (inclusive) -> End
    """
    kernels = block_data['kernels']
    layer_idx = block_data['layer']
    
    t_p1, t_p2, t_p3, t_p4 = 0.0, 0.0, 0.0, 0.0
    
    # state 0: Attn_proj
    # state 1: KV_cache (post-rope)
    # state 2: Attn_core (post-flash_attn)
    # state 3: FFN_block (post-rms_norm for FFN)
    state = 0 
    
    for k in kernels:
        name = k['name']
        dur = k['dur']
        
        if state == 0:
            if "rope" in name:
                state = 1
                t_p2 += dur # rope 归入 KV_cache
            else:
                t_p1 += dur
                
        elif state == 1:
            if "flash_attn" in name:
                state = 2
                t_p3 += dur # flash_attn 归入 Attn_core
            else:
                t_p2 += dur
                
        elif state == 2:
            # 此时在 Attn_core 阶段 (FlashAttn 之后，FFN 之前)
            # 我们需要寻找进入 FFN 的标志：FFN 前的 RMS Norm
            # 注意：FlashAttn 后可能紧跟一个 mul (Attn Output Proj) 和一个可选的 add (Resid)
            # 这里的 rms_norm 是区分关键
            if "kernel_rms_norm" in name:
                state = 3
                t_p4 += dur # rms_norm 归入 FFN_block
            else:
                t_p3 += dur
                
        elif state == 3:
            # FFN 阶段直到 Block 结束
            t_p4 += dur
            
    return {
        'layer': layer_idx,
        'times': [t_p1, t_p2, t_p3, t_p4]
    }

def print_table(title, data_dict):
    print(f"\n{'='*20} {title} {'='*20}")
    headers = ["Layer", "Attn_proj (us)", "KV_cache (us)", "Attn_core (us)", "FFN_block (us)", "Total (us)"]
    print(f"{headers[0]:<6} | {headers[1]:<14} | {headers[2]:<14} | {headers[3]:<14} | {headers[4]:<14} | {headers[5]:<10}")
    print("-" * 85)
    
    total_avg_p1 = 0
    total_avg_p2 = 0
    total_avg_p3 = 0
    total_avg_p4 = 0
    
    sorted_layers = sorted(data_dict.keys())
    for layer in sorted_layers:
        entries = data_dict[layer]
        count = len(entries)
        if count == 0: continue
        
        avg_p1 = sum(x[0] for x in entries) / count
        avg_p2 = sum(x[1] for x in entries) / count
        avg_p3 = sum(x[2] for x in entries) / count
        avg_p4 = sum(x[3] for x in entries) / count
        
        total_avg_p1 += avg_p1
        total_avg_p2 += avg_p2
        total_avg_p3 += avg_p3
        total_avg_p4 += avg_p4
        
        layer_total = avg_p1 + avg_p2 + avg_p3 + avg_p4
        print(f"{layer:<6} | {avg_p1:>14.2f} | {avg_p2:>14.2f} | {avg_p3:>14.2f} | {avg_p4:>14.2f} | {layer_total:>10.2f}")

    print("-" * 85)
    token_total = total_avg_p1 + total_avg_p2 + total_avg_p3 + total_avg_p4
    print(f"{'TOKEN':<6} | {total_avg_p1:>14.2f} | {total_avg_p2:>14.2f} | {total_avg_p3:>14.2f} | {total_avg_p4:>14.2f} | {token_total:>10.2f}")

def main():
    # === 新增命令行解析逻辑 ===
    parser = argparse.ArgumentParser(
        description="LLM Trace Analysis Tool - 自动解析并统计 GPU Kernel 阶段耗时",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 位置参数：文件路径
    parser.add_argument(
        "file_path", 
        help="输入的 JSON trace 文件路径"
    )
    
    # 可选参数：导出结果到文件
    parser.add_argument(
        "-o", "--output", 
        help="将分析结果保存到指定的文本文件", 
        metavar="FILE"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.file_path):
        print(f"错误: 找不到文件 '{args.file_path}'")
        sys.exit(1)

    # 如果指定了输出文件，重定向 stdout
    original_stdout = sys.stdout
    output_file = None
    if args.output:
        output_file = open(args.output, 'w', encoding='utf-8')
        sys.stdout = output_file

    try:
        print(f"正在处理 {args.file_path} ...")
        kernels = load_and_clean_trace(args.file_path)
        
        if not kernels:
            print("未找到 Device Kernel。")
            return

        print(f"找到 {len(kernels)} 个 Device Kernel。正在进行 Block 切分...")
        raw_blocks = analyze_blocks(kernels)
        
        if not raw_blocks:
            print("警告: 未能识别到任何有效的 Layer Block，请检查算子命名规则。")
            return
            
        print(f"识别到 {len(raw_blocks)} 个 Block。")
        
        stats = { "Prefill": {}, "Decode": {} }
        current_pass_mode = "Decode"
        
        for raw in raw_blocks:
            layer = raw['layer']
            # 模式检测逻辑
            if layer < 27:
                has_mm = any("kernel_mul_mm" in k['name'] for k in raw['kernels'])
                current_pass_mode = "Prefill" if has_mm else "Decode"
            
            res = process_single_block(raw)
            
            mode_dict = stats[current_pass_mode]
            if layer not in mode_dict:
                mode_dict[layer] = []
            mode_dict[layer].append(res['times'])
            
        if stats["Prefill"]:
            print_table("Prefill Phase Analysis", stats["Prefill"])
        
        if stats["Decode"]:
            print_table("Decode Phase Analysis", stats["Decode"])
            
    finally:
        # 恢复 stdout 并关闭文件
        if output_file:
            sys.stdout = original_stdout
            output_file.close()
            print(f"分析完成！结果已保存至: {args.output}")

if __name__ == "__main__":
    main()