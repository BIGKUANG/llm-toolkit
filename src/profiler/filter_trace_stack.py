#!/usr/bin/env python3
"""
Chrome Trace JSON 堆栈深度过滤工具

用于后处理 PyTorch Profiler 导出的 trace JSON 文件，
过滤堆栈跟踪的深度以减少文件大小和提高可读性。
支持 X 轴时间范围截取。
支持按操作名称过滤并提取上下调用栈。

使用方法:
    python filter_trace_stack.py <input_file> [--output_file OUTPUT] [--depth N]
                                 [--trim_start_percent P] [--trim_end_percent P]
                                 [--filter_op OP_NAME]

参数:
    input_file          : 输入 trace JSON 文件路径
    --output_file, -o   : 输出 trace JSON 文件路径 (可选，默认在输入文件同目录添加 _filtered 后缀)
    --depth, -d         : 最大堆栈深度 (可选，默认不进行堆栈深度过滤)
    --trim_start_percent: 从左往右截取百分比，保留后 (100-P)% 的数据 (默认: 0)
    --trim_end_percent  : 从右往左截取百分比，保留前 (100-P)% 的数据 (默认: 0)
    --filter_op         : 过滤操作名称，只保留匹配的事件及其上下调用栈 (可选)

示例:
    python filter_trace_stack.py logs/empty_test_trace.json --depth 5
    python filter_trace_stack.py trace.json --output_file trace_filtered.json -d 3
    python filter_trace_stack.py trace.json --trim_start_percent 10 --trim_end_percent 20
    python filter_trace_stack.py trace.json --filter_op "aten::to"
"""

import argparse
import json
import os
from typing import Any


def filter_stack_depth(stack_str: str, max_depth: int) -> str:
    """
    过滤堆栈字符串深度
    
    Args:
        stack_str: 原始堆栈字符串，通常以换行符分隔
        max_depth: 最大保留深度
        
    Returns:
        过滤后的堆栈字符串
    """
    if not stack_str:
        return stack_str
    
    lines = stack_str.split('\n')
    filtered_lines = lines[:max_depth]
    return '\n'.join(filtered_lines)


def process_event(event: dict[str, Any], max_depth: int) -> dict[str, Any]:
    """
    处理单个事件，过滤堆栈深度
    
    Args:
        event: trace 事件字典
        max_depth: 最大保留深度
        
    Returns:
        处理后的事件字典
    """
    # 处理 data 字段中的堆栈信息
    if 'data' in event and isinstance(event['data'], dict):
        data = event['data']
        
        # 过滤 stack 字段
        if 'stack' in data and isinstance(data['stack'], str):
            data['stack'] = filter_stack_depth(data['stack'], max_depth)
        
        # 过滤 pythonTrace 字段（如果存在）
        if 'pythonTrace' in data and isinstance(data['pythonTrace'], str):
            data['pythonTrace'] = filter_stack_depth(data['pythonTrace'], max_depth)
    
    return event


def get_time_range(events: list[dict[str, Any]]) -> tuple[float, float]:
    """
    获取事件列表的时间范围
    
    Args:
        events: 事件列表
        
    Returns:
        (最小时间戳, 最大时间戳)
    """
    timestamps = []
    for event in events:
        if 'ts' in event and isinstance(event['ts'], (int, float)):
            timestamps.append(event['ts'])
    
    if not timestamps:
        return (0, 0)
    
    return (min(timestamps), max(timestamps))


def filter_events_by_time_range(
    events: list[dict[str, Any]],
    trim_start_percent: float,
    trim_end_percent: float
) -> list[dict[str, Any]]:
    """
    根据时间范围百分比过滤事件
    
    Args:
        events: 事件列表
        trim_start_percent: 从左往右截取百分比 (0-100)
        trim_end_percent: 从右往左截取百分比 (0-100)
        
    Returns:
        过滤后的事件列表
    """
    if trim_start_percent == 0 and trim_end_percent == 0:
        return events
    
    # 获取时间范围
    min_ts, max_ts = get_time_range(events)
    if min_ts == max_ts:
        print("警告: 所有事件时间戳相同，无法进行时间范围过滤")
        return events
    
    total_duration = max_ts - min_ts
    
    # 计算保留的时间范围
    start_offset = total_duration * (trim_start_percent / 100)
    end_offset = total_duration * (trim_end_percent / 100)
    
    keep_start_ts = min_ts + start_offset
    keep_end_ts = max_ts - end_offset
    
    if keep_start_ts >= keep_end_ts:
        print("警告: 截取范围无效，保留的时间范围为空")
        return []
    
    print(f"时间范围: {min_ts:.2f} - {max_ts:.2f} (总时长: {total_duration:.2f})")
    print(f"保留范围: {keep_start_ts:.2f} - {keep_end_ts:.2f}")
    
    # 过滤事件
    filtered_events = []
    for event in events:
        # 保留没有时间戳的事件（如 metadata 事件）
        if 'ts' not in event:
            filtered_events.append(event)
            continue
        
        ts = event['ts']
        if isinstance(ts, (int, float)) and keep_start_ts <= ts <= keep_end_ts:
            filtered_events.append(event)
    
    return filtered_events


def get_event_time_span(event: dict[str, Any]) -> tuple[float, float] | None:
    """
    获取事件的时间范围
    
    Args:
        event: 事件字典
        
    Returns:
        (开始时间, 结束时间) 或 None（如果没有时间戳）
    """
    if 'ts' not in event or not isinstance(event['ts'], (int, float)):
        return None
    
    start_ts = event['ts']
    
    # 如果有持续时间，计算结束时间
    if 'dur' in event and isinstance(event['dur'], (int, float)):
        end_ts = start_ts + event['dur']
    else:
        # 没有持续时间的事件，使用开始时间作为结束时间
        end_ts = start_ts
    
    return (start_ts, end_ts)


def is_time_overlap(span1: tuple[float, float], span2: tuple[float, float]) -> bool:
    """
    判断两个时间范围是否有重叠
    
    Args:
        span1: (开始时间, 结束时间)
        span2: (开始时间, 结束时间)
        
    Returns:
        是否有重叠
    """
    return span1[0] <= span2[1] and span2[0] <= span1[1]


def is_contained(inner_span: tuple[float, float], outer_span: tuple[float, float]) -> bool:
    """
    判断 inner_span 是否完全包含在 outer_span 内
    
    Args:
        inner_span: 内部时间范围
        outer_span: 外部时间范围
        
    Returns:
        是否包含
    """
    return outer_span[0] <= inner_span[0] and inner_span[1] <= outer_span[1]


def find_matching_events(events: list[dict[str, Any]], op_name: str) -> list[dict[str, Any]]:
    """
    查找匹配指定操作名称的事件
    
    Args:
        events: 事件列表
        op_name: 操作名称（支持部分匹配）
        
    Returns:
        匹配的事件列表
    """
    matching_events = []
    for event in events:
        if 'name' in event and op_name in event['name']:
            matching_events.append(event)
    return matching_events


def get_event_duration(event: dict[str, Any]) -> float | None:
    """
    获取事件的持续时间
    
    Args:
        event: 事件字典
        
    Returns:
        持续时间或 None（如果没有持续时间）
    """
    if 'dur' in event and isinstance(event['dur'], (int, float)):
        return event['dur']
    return None


def merge_time_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    合并重叠的时间范围
    
    Args:
        spans: 时间范围列表 [(start, end), ...]
        
    Returns:
        合并后的时间范围列表
    """
    if not spans:
        return []
    
    # 按开始时间排序
    sorted_spans = sorted(spans, key=lambda x: x[0])
    
    merged = [sorted_spans[0]]
    for current_start, current_end in sorted_spans[1:]:
        last_start, last_end = merged[-1]
        
        # 如果当前范围与最后一个合并范围重叠或相邻，则合并
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    
    return merged


def is_in_merged_spans(ts: float, merged_spans: list[tuple[float, float]]) -> bool:
    """
    检查时间戳是否在合并后的时间范围内（使用二分查找）
    
    Args:
        ts: 时间戳
        merged_spans: 合并后的时间范围列表（已按开始时间排序）
        
    Returns:
        是否在范围内
    """
    if not merged_spans:
        return False
    
    # 使用二分查找定位可能的时间范围
    left, right = 0, len(merged_spans) - 1
    
    while left <= right:
        mid = (left + right) // 2
        span_start, span_end = merged_spans[mid]
        
        if ts < span_start:
            right = mid - 1
        elif ts > span_end:
            left = mid + 1
        else:
            return True
    
    return False


def filter_events_by_op_name(
    events: list[dict[str, Any]],
    op_name: str,
    enable_least: bool = False
) -> list[dict[str, Any]]:
    """
    根据操作名称过滤事件，只保留匹配的事件及其时间范围内的子事件
    
    Args:
        events: 事件列表
        op_name: 操作名称
        enable_least: 是否只选择耗时最少的操作
        
    Returns:
        过滤后的事件列表
    """
    # 查找匹配的事件
    matching_events = find_matching_events(events, op_name)
    
    if not matching_events:
        print(f"警告: 未找到匹配 '{op_name}' 的操作")
        return events
    
    print(f"找到 {len(matching_events)} 个匹配 '{op_name}' 的操作")
    
    # 如果启用 enable_least，只选择耗时最少的操作
    if enable_least:
        # 过滤出有持续时间的操作
        events_with_dur = [(e, get_event_duration(e)) for e in matching_events]
        events_with_dur = [(e, d) for e, d in events_with_dur if d is not None]
        
        if not events_with_dur:
            print("警告: 匹配的事件没有持续时间信息，无法选择耗时最少的操作")
            # 使用第一个匹配的事件
            selected_events = [matching_events[0]]
        else:
            # 选择耗时最少的操作
            min_event = min(events_with_dur, key=lambda x: x[1])
            selected_events = [min_event[0]]
            print(f"选择耗时最少的操作: 持续时间 {min_event[1]:.2f} us")
    else:
        selected_events = matching_events
    
    # 获取选中事件的时间范围
    target_time_spans = []
    for event in selected_events:
        span = get_event_time_span(event)
        if span is not None:
            target_time_spans.append(span)
    
    if not target_time_spans:
        print("警告: 选中的事件没有有效的时间戳")
        return events
    
    # 合并重叠的时间范围，提高查询效率
    merged_spans = merge_time_spans(target_time_spans)
    print(f"合并后时间范围数量: {len(merged_spans)} (原始: {len(target_time_spans)})")
    
    # 过滤事件：只保留在目标时间范围内的事件
    filtered_events = []
    for event in events:
        # 保留没有时间戳的事件（如 metadata 事件）
        if 'ts' not in event:
            filtered_events.append(event)
            continue
        
        ts = event['ts']
        if not isinstance(ts, (int, float)):
            filtered_events.append(event)
            continue
        
        # 获取事件结束时间
        dur = event.get('dur', 0)
        if not isinstance(dur, (int, float)):
            dur = 0
        end_ts = ts + dur
        
        # 检查事件是否与任何合并后的时间范围有重叠
        # 使用二分查找优化
        for span_start, span_end in merged_spans:
            if ts <= span_end and end_ts >= span_start:
                filtered_events.append(event)
                break
    
    print(f"时间范围过滤后共 {len(filtered_events)} 个事件")
    
    return filtered_events


def filter_trace_file(
    input_file: str,
    output_file: str,
    max_depth: int | None,
    trim_start_percent: float,
    trim_end_percent: float,
    filter_op: str | None,
    enable_least: bool = False
) -> None:
    """
    过滤 trace JSON 文件的堆栈深度和时间范围
    
    Args:
        input_file: 输入 trace JSON 文件路径
        output_file: 输出 trace JSON 文件路径
        max_depth: 最大保留堆栈深度（可选）
        trim_start_percent: 从左往右截取百分比
        trim_end_percent: 从右往左截取百分比
        filter_op: 过滤操作名称
        enable_least: 是否只选择耗时最少的操作
    """
    print(f"正在读取文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        trace_data = json.load(f)
    
    events = None
    events_key = None
    
    # 处理不同格式的 trace 数据
    if isinstance(trace_data, dict):
        # PyTorch Profiler 格式
        if 'traceEvents' in trace_data:
            events = trace_data['traceEvents']
            events_key = 'traceEvents'
        # 其他可能的字典格式
        elif 'events' in trace_data:
            events = trace_data['events']
            events_key = 'events'
        else:
            print("警告: 未识别的 trace 格式")
    
    elif isinstance(trace_data, list):
        # 直接是事件列表
        events = trace_data
        events_key = None
    else:
        print("警告: 未识别的 trace 格式")
    
    if events is None:
        print("错误: 无法找到事件数据")
        return
    
    total_events = len(events)
    print(f"共发现 {total_events} 个事件")
    
    # 1. 先进行操作名称过滤
    if filter_op is not None:
        events = filter_events_by_op_name(events, filter_op, enable_least)
    
    # 2. 再进行时间范围过滤
    if trim_start_percent > 0 or trim_end_percent > 0:
        events = filter_events_by_time_range(events, trim_start_percent, trim_end_percent)
        print(f"时间过滤后剩余 {len(events)} 个事件")
    
    # 3. 最后进行堆栈深度过滤（仅在指定深度时）
    if max_depth is not None:
        for i, event in enumerate(events):
            events[i] = process_event(event, max_depth)
    
    # 更新 trace 数据
    if events_key is not None:
        trace_data[events_key] = events
    else:
        trace_data = events
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 写入输出文件
    print(f"正在写入文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trace_data, f, ensure_ascii=False, separators=(',', ':'))
    
    # 计算文件大小变化
    input_size = os.path.getsize(input_file)
    output_size = os.path.getsize(output_file)
    reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0
    
    print(f"处理完成!")
    print(f"原始文件大小: {input_size / 1024 / 1024:.2f} MB")
    print(f"处理后文件大小: {output_size / 1024 / 1024:.2f} MB")
    print(f"文件大小减少: {reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Chrome Trace JSON 堆栈深度和时间范围过滤工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python filter_trace_stack.py input.json --depth 10
  python filter_trace_stack.py trace.json --output_file trace_filtered.json -d 5
  python filter_trace_stack.py trace.json --trim_start_percent 10 --trim_end_percent 20
  python filter_trace_stack.py trace.json -d 5 --trim_start_percent 10
  python filter_trace_stack.py trace.json --filter_op "aten::to"
  python filter_trace_stack.py trace.json --filter_op "aten::to" -d 5
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入 trace JSON 文件路径'
    )
    
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        default=None,
        help='输出 trace JSON 文件路径 (可选，默认在输入文件同目录添加 _filtered 后缀)'
    )
    
    parser.add_argument(
        '-d', '--depth',
        type=int,
        default=None,
        help='最大堆栈深度 (可选，默认不进行堆栈深度过滤)'
    )
    
    parser.add_argument(
        '--trim_start_percent',
        type=float,
        default=0,
        help='从左往右截取百分比，保留后 (100-P)% 的数据 (默认: 0)'
    )
    
    parser.add_argument(
        '--trim_end_percent',
        type=float,
        default=0,
        help='从右往左截取百分比，保留前 (100-P)% 的数据 (默认: 0)'
    )
    
    parser.add_argument(
        '--filter_op',
        type=str,
        default=None,
        help='过滤操作名称，只保留匹配的事件及其时间范围内的子事件 (可选，支持部分匹配)'
    )
    
    parser.add_argument(
        '--enable_least',
        action='store_true',
        help='与 --filter_op 配合使用，只选择耗时最少的操作进行导出'
    )
    
    args = parser.parse_args()
    
    # 验证输入文件存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        return
    
    # 验证深度参数
    if args.depth is not None and args.depth < 1:
        print(f"错误: 深度必须大于 0，当前值: {args.depth}")
        return
    
    # 验证截取百分比参数
    if args.trim_start_percent < 0 or args.trim_start_percent > 100:
        print(f"错误: trim_start_percent 必须在 0-100 之间，当前值: {args.trim_start_percent}")
        return
    
    if args.trim_end_percent < 0 or args.trim_end_percent > 100:
        print(f"错误: trim_end_percent 必须在 0-100 之间，当前值: {args.trim_end_percent}")
        return
    
    if args.trim_start_percent + args.trim_end_percent >= 100:
        print(f"错误: trim_start_percent + trim_end_percent 必须小于 100")
        return
    
    # 如果未指定输出文件，自动生成带后缀的文件名
    output_file = args.output_file
    if output_file is None:
        # 获取文件名和扩展名
        base_name, ext = os.path.splitext(args.input_file)
        suffix_parts = []
        if args.depth is not None:
            suffix_parts.append(f"d{args.depth}")
        if args.filter_op is not None:
            # 简化操作名称用于文件名
            op_suffix = args.filter_op.replace("::", "_").replace(" ", "_")
            suffix_parts.append(f"op_{op_suffix}")
        if args.trim_start_percent > 0:
            suffix_parts.append(f"ts{int(args.trim_start_percent)}")
        if args.trim_end_percent > 0:
            suffix_parts.append(f"te{int(args.trim_end_percent)}")
        if suffix_parts:
            output_file = f"{base_name}_filtered_{','.join(suffix_parts)}{ext}"
        else:
            output_file = f"{base_name}_filtered{ext}"
    
    filter_trace_file(
        args.input_file,
        output_file,
        args.depth,
        args.trim_start_percent,
        args.trim_end_percent,
        args.filter_op,
        args.enable_least
    )


if __name__ == '__main__':
    main()