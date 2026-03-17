# OpTracer - 算子参数追踪工具

用于追踪和记录算子/函数调用时的参数信息，支持多种输出格式。

## 安装

```bash
# 确保 llm_tools 在 Python 路径中
import sys
sys.path.insert(0, "/path/to/llm_tools")
```

## 快速开始

### 方式1: 使用装饰器（推荐）

```python
from llm_tools.trace.op_tracer import trace_op, start_trace, stop_trace

# 使用 @trace_op 装饰器标记要追踪的函数
@trace_op
def my_operator(a, b):
    return a + b

# 启动追踪
start_trace()

# 调用函数（会被自动追踪）
result = my_operator(tensor_a, tensor_b)

# 停止追踪
stop_trace()
```

### 方式2: 使用上下文管理器

```python
from llm_tools.trace.op_tracer import trace_op, tracing

@trace_op
def my_operator(a, b):
    return a + b

# 使用 with 语句自动管理追踪生命周期
with tracing():
    result = my_operator(tensor_a, tensor_b)
    # 退出 with 块后自动停止追踪
```

### 方式3: 包装现有函数

```python
from llm_tools.trace.op_tracer import trace_op, start_trace, stop_trace
import flag_gems

# 包装已有的函数
traced_index_put = trace_op(flag_gems.index_put)

start_trace(output_dir="/tmp/my_trace")
result = traced_index_put(input_tensor, indices, values)
stop_trace()
```

### 方式4: 自定义 Tracer 实例

```python
from llm_tools.trace.op_tracer import OpTracer

# 创建自定义 tracer
tracer = OpTracer(
    output_dir="/custom/path",
    print_to_console=False,
    enable_dedup=True,
)

# 使用上下文管理器
with tracer.session():
    @tracer.trace
    def my_op(x):
        return x * 2
    
    my_op(tensor)
```

## API 参考

### 全局函数

| 函数 | 说明 |
|------|------|
| `trace_op(func)` | 装饰器：追踪函数调用 |
| `start_trace(output_dir=None, print_to_console=True)` | 启动全局追踪 |
| `stop_trace()` | 停止全局追踪 |
| `is_tracing()` | 检查是否正在追踪 |
| `tracing(output_dir=None, print_to_console=True)` | 上下文管理器 |
| `get_tracer()` | 获取默认 tracer 实例 |

### OpTracer 类

```python
class OpTracer:
    def __init__(
        self,
        print_to_console: bool = True,      # 是否输出到控制台
        log_file: str = None,               # 日志文件路径
        jsonl_file: str = None,             # JSONL 文件路径
        csv_file: str = None,               # CSV 文件路径
        output_dir: str = None,             # 输出目录（自动生成文件路径）
        enable_dedup: bool = True,          # 是否启用去重（仅文件输出）
        marker_start: str = "<<<OP_TRACE_START>>>",
        marker_end: str = "<<<OP_TRACE_END>>>",
        auto_start: bool = False,           # 是否自动启动
    )
    
    def start(self)          # 启动追踪
    def stop(self)           # 停止追踪
    def is_started(self)     # 检查是否已启动
    def session(self)        # 上下文管理器
    def trace(func)          # 装饰器
```

## 输出格式

### 控制台输出

```
<<<OP_TRACE_START>>>
[TIMESTAMP] 2024-01-15T10:30:00.123456
[OPERATOR] index_put
[ARGS]
  type: tuple[3]
  [0]:
    type: tensor
    shape: [256, 256]
    dtype: torch.bfloat16
    device: cuda:0
    requires_grad: False
    stride: [256, 1]
  ...
<<<OP_TRACE_END>>>
```

### JSONL 文件

```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "operator": "index_put",
  "args": {
    "type": "tuple[3]",
    "items": [...]
  }
}
```

### CSV 文件

自动扁平化参数信息，方便导入 Excel 或数据分析工具。

## 去重功能

- **控制台输出**：从不去重，每次调用都会打印
- **文件输出**：默认启用去重，相同参数的重复调用只记录一次

可以通过 `enable_dedup=False` 禁用去重：

```python
tracer = OpTracer(enable_dedup=False)
```

## 完整示例

```python
import torch
from llm_tools.trace.op_tracer import trace_op, start_trace, stop_trace

# 定义要追踪的算子
@trace_op
def custom_matmul(a, b, bias=None):
    result = torch.matmul(a, b)
    if bias is not None:
        result = result + bias
    return result

# 启动追踪（输出到自定义目录）
start_trace(output_dir="/tmp/my_model_trace")

# 创建测试数据
a = torch.randn(100, 256, device='cuda')
b = torch.randn(256, 512, device='cuda')
bias = torch.randn(512, device='cuda')

# 调用函数（自动追踪）
result = custom_matmul(a, b, bias=bias)

# 停止追踪
stop_trace()

# 查看输出文件
# /tmp/my_model_trace/op_trace.jsonl
# /tmp/my_model_trace/op_trace.log
# /tmp/my_model_trace/op_trace.csv
```

## 向后兼容

以下名称已保留用于向后兼容：

- `index_put_tracer` - 默认 tracer 实例的别名
- `default_tracer` - 默认 tracer 实例的别名
- `print_args` - OpTracer 类的别名

## 更新日志

### v2.0.0 (2026-03-14)

**重大重构：**

1. **移除 `_LazyTracer` 类** - 不再需要复杂的延迟查找机制
2. **新增全局便捷函数**：
   - `trace_op()` - 装饰器
   - `start_trace()` / `stop_trace()` - 控制追踪
   - `is_tracing()` - 检查状态
   - `tracing()` - 上下文管理器
3. **简化使用方式**：
   - 不需要手动创建 tracer 实例
   - 不需要导入 `index_put_tracer`
   - 可以直接包装现有函数
4. **改进的 API 设计**：
   - 单例模式的默认 tracer
   - 更清晰的函数命名
   - 更好的类型提示

**迁移指南：**

```python
# 旧方式
from llm_tools.trace.op_tracer import index_put_tracer
index_put_tracer.start()
# ... 调用被装饰的函数
index_put_tracer.stop()

# 新方式
from llm_tools.trace.op_tracer import start_trace, stop_trace
start_trace()
# ... 调用被装饰的函数
stop_trace()
```