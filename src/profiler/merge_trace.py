#!/usr/bin/env python3
"""
PyTorch Profiler Trace 合并工具

用于合并多个 rank 的 trace JSON 文件，方便在 Chrome Tracing 中查看。

使用方法:
    python merge_trace.py <input_dir> [--output_file OUTPUT]

参数:
    input_dir   : 包含 trace JSON 文件的目录
    --output_file, -o : 输出文件路径 (可选，默认保存到输入目录下的 merged_trace.json)

示例:
    # 合并 profile 目录下的所有 trace 文件
    python merge_trace.py ./profile
    
    # 指定输出文件
    python merge_trace.py ./profile -o merged.json
    
    # 合并当前目录下的 trace 文件
    python merge_trace.py .
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


class TraceMerger:
    """Trace 文件合并器"""
    
    def __init__(self, input_dir: str, output_file: str | None = None):
        """
        初始化合并器
        
        Args:
            input_dir: 输入目录
            output_file: 输出文件路径
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file) if output_file else None
        
    def find_trace_files(self) -> list[Path]:
        """查找目录下的所有 trace JSON 文件"""
        trace_files = []
        torch_trace_files = sorted(list(self.input_dir.glob("*.json*")))
        trace_files = torch_trace_files
        # runtime_trace_files = sorted(list(self.input_dir.glob("*.out*")))
        # trace_files = runtime_trace_files

        # 按文件名排序，确保 rank 顺序正确
        # for t_f,  r_f in zip(torch_trace_files, runtime_trace_files):
        #     trace_files.append(t_f)
        #     trace_files.append(r_f)

        # trace_files.extend(torch_trace_files[4:6])
        # trace_files.extend(runtime_trace_files[4:6])
        print("===> trace_files: \n", trace_files)
        print("\n")
        # assert 0
        return trace_files
    
    def extract_rank(self, filename: str) -> int:
        """
        从文件名中提取 rank 编号
        
        Args:
            filename: 文件名
            
        Returns:
            rank 编号，如果无法提取则返回文件在列表中的索引
        """
        # 尝试匹配 rank_N 格式
        match = re.search(r'rank[_\-]?(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return -1
    
    def load_trace(self, filepath: Path) -> dict[str, Any]:
        """加载 trace 文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def merge_traces(self, trace_files: list[Path]) -> dict[str, Any]:
        """
        合并多个 trace 文件
        
        Args:
            trace_files: trace 文件列表
            
        Returns:
            合并后的 trace 数据
        """
        if not trace_files:
            raise ValueError("没有找到 trace 文件")
        
        # 使用第一个文件作为基础
        base_trace = self.load_trace(trace_files[0])
        merged_events = list(base_trace.get('traceEvents', []))
        
        # 为第一个 rank 的事件添加 rank 标识
        rank = self.extract_rank(trace_files[0].name)
        self._add_rank_to_events(merged_events, rank if rank >= 0 else 0)
        
        # 合并其他 rank 的 trace
        for i, filepath in enumerate(trace_files):
            try:
                trace_data = self.load_trace(filepath)
            except Exception as e:
                print(f"捕获到一个异常: {type(e).__name__} - {e}")
                print("===> load filepath: ", filepath)
                continue

            events = trace_data.get('traceEvents', [])
            
            # 添加 rank 标识
            rank = self.extract_rank(filepath.name)
            self._add_rank_to_events(events, rank if rank >= 0 else i)
            
            merged_events.extend(events)
        
        # 更新合并后的 trace
        base_trace['traceEvents'] = merged_events
        
        # 更新 trace 名称
        base_trace['traceName'] = f"Merged Trace ({len(trace_files)} ranks)"
        
        return base_trace
    
    def _add_rank_to_events(self, events: list[dict], rank: int) -> None:
        """
        为事件添加 rank 标识
        
        通过修改 pid 来区分不同 rank，这样在 Chrome Tracing 中会显示为不同的进程行
        
        Args:
            events: 事件列表
            rank: rank 编号
        """
        for event in events:
            # 在 args 中添加 rank 信息
            if 'args' not in event:
                event['args'] = {}
            event['args']['rank'] = rank
            
            # 修改 pid 以区分不同 rank（Chrome Tracing 会按 pid 分组显示）
            if 'pid' in event:
                # 使用 rank + 1 作为新 pid，避免 pid 为 0
                event['pid'] = f"rank_{rank}"
    
    def run(self) -> str:
        """
        执行合并
        
        Returns:
            输出文件路径
        """
        # 查找 trace 文件
        trace_files = self.find_trace_files()
        
        if not trace_files:
            raise ValueError(f"在 {self.input_dir} 中没有找到 JSON 文件")
        
        print(f"找到 {len(trace_files)} 个 trace 文件:")
        for f in trace_files:
            print(f"  - {f.name}")
        
        # 合并 trace
        print("\n正在合并...")
        merged_trace = self.merge_traces(trace_files)
        
        # 确定输出路径
        if self.output_file is None:
            # 默认保存到当前工作目录
            self.output_file = Path.cwd() / "merged_trace.json"
        else:
            # 确保输出目录存在
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_trace, f, ensure_ascii=False, separators=(',', ':'))
        
        # 统计信息
        total_events = len(merged_trace.get('traceEvents', []))
        output_size = self.output_file.stat().st_size / (1024 * 1024)
        
        print(f"\n合并完成!")
        print(f"总事件数: {total_events}")
        print(f"输出文件: {self.output_file}")
        print(f"文件大小: {output_size:.2f} MB")
        
        return str(self.output_file)


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Profiler Trace 合并工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python merge_trace.py ./profile
  python merge_trace.py ./profile -o merged.json
  python merge_trace.py . -o output/merged_trace.json
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='包含 trace JSON 文件的目录'
    )
    
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        default=None,
        help='输出文件路径 (可选，默认保存到输入目录下的 merged_trace.json)'
    )
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 执行合并
    merger = TraceMerger(args.input_dir, args.output_file)
    merger.run()


if __name__ == '__main__':
    main()