"""
算子参数追踪工具 (Operator Tracer)

用于追踪和记录算子/函数调用时的参数信息。
支持多种输出格式：console, log, jsonl, csv
支持动态启停控制
"""
import json
import hashlib
import csv
import os
import atexit
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, TextIO
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from pathlib import Path
import torch


class OpTracer:
    """算子参数追踪器
    
    特性:
    - 自动获取算子名称（从被装饰函数）
    - 支持多种输出格式: console, log, jsonl, csv
    - 支持动态启停控制 (start/stop)
    - console输出不去重，文件输出去重
    - 所有文件格式支持逐行实时保存
    """
    
    _global_started: bool = False
    _global_tracers: Set['OpTracer'] = set()
    
    def __init__(
        self,
        print_to_console: bool = True,
        log_file: Optional[str] = None,
        jsonl_file: Optional[str] = None,
        csv_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        enable_dedup: bool = True,
        marker_start: str = "<<<OP_TRACE_START>>>",
        marker_end: str = "<<<OP_TRACE_END>>>",
        auto_start: bool = False,
    ):
        self.print_to_console = print_to_console
        self.enable_dedup = enable_dedup
        self.marker_start = marker_start
        self.marker_end = marker_end
        self.output_dir = output_dir
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            if jsonl_file is None:
                jsonl_file = os.path.join(output_dir, "op_trace.jsonl")
            if log_file is None:
                log_file = os.path.join(output_dir, "op_trace.log")
            if csv_file is None:
                csv_file = os.path.join(output_dir, "op_trace.csv")
        
        self.log_file = log_file
        self.jsonl_file = jsonl_file
        self.csv_file = csv_file
        
        self._started: bool = auto_start
        self._seen_hashes: Set[str] = set()
        self._jsonl_fp: Optional[TextIO] = None
        self._csv_fp: Optional[TextIO] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fieldnames: List[str] = []
        
        OpTracer._global_tracers.add(self)
        atexit.register(self._cleanup)
        
        if auto_start:
            self.start()
    
    def start(self):
        """启动追踪功能"""
        self._started = True
        OpTracer._global_started = True
        # 如果文件已关闭，重新初始化
        if self._jsonl_fp is None or self._jsonl_fp.closed:
            self._init_files()
        return self
    
    def stop(self):
        """停止追踪功能并关闭文件"""
        self._started = False
        self._close_files()
        if not any(t._started for t in OpTracer._global_tracers):
            OpTracer._global_started = False
        return self
    
    def is_started(self) -> bool:
        return self._started
    
    @contextmanager
    def session(self):
        """上下文管理器，自动启动和停止追踪"""
        self.start()
        try:
            yield self
        finally:
            self.stop()
    
    def _init_files(self):
        # 重新打开 JSONL 文件
        if self.jsonl_file:
            if self._jsonl_fp is None or self._jsonl_fp.closed:
                Path(self.jsonl_file).parent.mkdir(parents=True, exist_ok=True)
                self._jsonl_fp = open(self.jsonl_file, 'a', encoding='utf-8')
        # 重新打开 CSV 文件
        if self.csv_file:
            if self._csv_fp is None or self._csv_fp.closed:
                Path(self.csv_file).parent.mkdir(parents=True, exist_ok=True)
                self._csv_fp = open(self.csv_file, 'a', newline='', encoding='utf-8')
                self._csv_writer = None  # 重置 writer
    
    def _close_files(self):
        if self._jsonl_fp:
            self._jsonl_fp.flush()
            self._jsonl_fp.close()
            self._jsonl_fp = None
        if self._csv_fp:
            self._csv_fp.flush()
            self._csv_fp.close()
            self._csv_fp = None
    
    def _cleanup(self):
        self._close_files()
    
    def trace(self, func: Callable) -> Callable:
        """装饰器函数 - 自动获取算子名称"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self._trace_op(func.__name__, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    
    def _trace_op(self, op_name: str, *args, **kwargs) -> bool:
        if not self._started:
            return False
        
        trace_info = {
            "timestamp": datetime.now().isoformat(),
            "operator": op_name,
            "args": self._get_value_info(args),
            "kwargs": {k: self._get_value_info(v) for k, v in kwargs.items()},
        }
        
        # 使用算子名称+输入shape进行去重
        dedup_key = self._compute_dedup_key(op_name, trace_info)
        is_duplicate = dedup_key in self._seen_hashes
        
        if self.print_to_console:
            self._output_to_console(op_name, trace_info)
        
        if self.enable_dedup and is_duplicate:
            return False
        
        self._seen_hashes.add(dedup_key)
        
        if self.log_file:
            self._output_to_log(op_name, trace_info)
        if self.jsonl_file:
            self._output_to_jsonl(trace_info)
        if self.csv_file:
            self._output_to_csv(trace_info)
        
        return True
    
    def _compute_dedup_key(self, op_name: str, trace_info: Dict) -> str:
        """计算去重键 - 基于算子名称+输入shape"""
        # 提取输入shape信息
        shapes = []
        args_info = trace_info.get('args', {})
        
        def extract_shapes(info):
            """递归提取shape信息"""
            if isinstance(info, dict):
                if info.get('type') == 'tensor':
                    return info.get('shape')
                elif 'items' in info:
                    return [extract_shapes(item) for item in info['items']]
            return None
        
        if 'items' in args_info:
            for item in args_info['items']:
                shape = extract_shapes(item)
                if shape:
                    shapes.append(str(shape))
        
        # 构建去重键: 算子名称 + shapes
        key_data = {
            "operator": op_name,
            "shapes": shapes
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _output_to_console(self, op_name: str, trace_info: Dict):
        lines = [self.marker_start, f"[TIMESTAMP] {trace_info['timestamp']}", f"[OPERATOR] {op_name}", "[ARGS]"]
        self._format_args(trace_info.get('args', {}), lines, 2)
        for k, v in trace_info.get('kwargs', {}).items():
            lines.append(f"  {k}:")
            self._format_item(v, lines, 4)
        lines.append(self.marker_end)
        print('\n'.join(lines), flush=True)
    
    def _format_args(self, info: Dict, lines: List[str], indent: int):
        if 'items' in info:
            for i, item in enumerate(info['items']):
                lines.append(f"{' ' * indent}args[{i}]:")
                self._format_item(item, lines, indent + 2)
    
    def _format_item(self, info: Any, lines: List[str], indent: int):
        p = ' ' * indent
        if isinstance(info, dict):
            if info.get('type') == 'tensor':
                lines.extend([f"{p}type: tensor", f"{p}shape: {info.get('shape')}", f"{p}dtype: {info.get('dtype')}", f"{p}device: {info.get('device')}"])
            elif 'value' in info:
                lines.extend([f"{p}type: {info.get('type')}", f"{p}value: {info['value']}"])
            elif 'items' in info:
                lines.append(f"{p}type: {info.get('type')}")
                for item in info['items']:
                    self._format_item(item, lines, indent + 2)
            else:
                lines.append(f"{p}{info}")
        else:
            lines.append(f"{p}{info}")
    
    def _output_to_log(self, op_name: str, trace_info: Dict):
        if not self.log_file:
            return
        lines = [self.marker_start, f"[TIMESTAMP] {trace_info['timestamp']}", f"[OPERATOR] {op_name}", "[ARGS]"]
        self._format_args(trace_info.get('args', {}), lines, 2)
        for k, v in trace_info.get('kwargs', {}).items():
            lines.append(f"  {k}:")
            self._format_item(v, lines, 4)
        lines.append(self.marker_end)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    
    def _output_to_jsonl(self, trace_info: Dict):
        if not self._jsonl_fp:
            return
        json.dump(trace_info, self._jsonl_fp, ensure_ascii=False, default=str)
        self._jsonl_fp.write('\n')
        self._jsonl_fp.flush()
    
    def _output_to_csv(self, trace_info: Dict):
        if not self._csv_fp:
            return
        flat = {'timestamp': trace_info.get('timestamp', ''), 'operator': trace_info.get('operator', '')}
        self._flatten_args(trace_info.get('args', {}), flat, 'arg')
        for k, v in trace_info.get('kwargs', {}).items():
            flat[f'kwarg_{k}_type'] = v.get('type', '')
            if 'value' in v:
                flat[f'kwarg_{k}_value'] = v['value']
        
        if self._csv_writer is None:
            self._csv_fieldnames = list(flat.keys())
            self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=self._csv_fieldnames)
            self._csv_writer.writeheader()
        
        self._csv_writer.writerow({f: flat.get(f, '') for f in self._csv_fieldnames})
        self._csv_fp.flush()
    
    def _flatten_args(self, info: Dict, result: Dict, prefix: str):
        if 'items' in info:
            for i, item in enumerate(info['items']):
                p = f"{prefix}_{i}_"
                result[f"{p}type"] = item.get('type', '')
                if item.get('type') == 'tensor':
                    result[f"{p}shape"] = str(item.get('shape', ''))
                    result[f"{p}dtype"] = str(item.get('dtype', ''))
                    result[f"{p}device"] = str(item.get('device', ''))
                elif 'value' in item:
                    result[f"{p}value"] = item['value']
    
    def _get_value_info(self, value: Any) -> Dict:
        if isinstance(value, torch.Tensor):
            return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device), "requires_grad": value.requires_grad}
        elif isinstance(value, (list, tuple)):
            return {"type": f"{type(value).__name__}[{len(value)}]", "items": [self._get_value_info(v) for v in value]}
        elif isinstance(value, dict):
            return {"type": f"dict[{len(value)}]", "items": {k: self._get_value_info(v) for k, v in value.items()}}
        else:
            return {"type": type(value).__name__, "value": value}
    
    def clear_dedup_cache(self):
        self._seen_hashes.clear()
    
    def __del__(self):
        self._close_files()
        if self in OpTracer._global_tracers:
            OpTracer._global_tracers.remove(self)


def create_tracer(name: str = "op_trace", output_dir: Optional[str] = None, enable_dedup: bool = True, print_to_console: bool = True, **kwargs) -> OpTracer:
    """创建预配置的OpTracer"""
    if output_dir is None:
        output_dir = f"/tmp/{name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return OpTracer(output_dir=output_dir, enable_dedup=enable_dedup, print_to_console=print_to_console, **kwargs)


# 预配置的 index_put 专用追踪器
index_put_tracer = OpTracer(
    print_to_console=True,
    output_dir="/tmp/index_put_trace",
    enable_dedup=True,
    marker_start="<<<INDEX_PUT_START>>>",
    marker_end="<<<INDEX_PUT_END>>>",
)


# 便捷函数
def trace_op(func: Callable) -> Callable:
    """便捷装饰器函数，使用 index_put_tracer"""
    return index_put_tracer.trace(func)


def is_tracing() -> bool:
    """检查 index_put_tracer 是否已启动"""
    return index_put_tracer.is_started()


def get_tracer() -> OpTracer:
    """获取默认的 tracer 实例"""
    return index_put_tracer


def start_trace(output_dir: Optional[str] = None, print_to_console: bool = True) -> OpTracer:
    """启动全局追踪
    
    Args:
        output_dir: 输出目录，默认使用 tracer 的默认目录
        print_to_console: 是否输出到控制台
    
    Returns:
        OpTracer 实例
    """
    tracer = index_put_tracer
    if output_dir:
        tracer.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    tracer.print_to_console = print_to_console
    tracer.start()
    return tracer


def stop_trace():
    """停止全局追踪"""
    index_put_tracer.stop()


@contextmanager
def tracing(output_dir: Optional[str] = None, print_to_console: bool = True):
    """上下文管理器，自动开始和停止追踪"""
    tracer = start_trace(output_dir, print_to_console)
    try:
        yield tracer
    finally:
        stop_trace()