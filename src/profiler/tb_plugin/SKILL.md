---
name: torch-tb-profiler-dev
description: PyTorch TensorBoard Profiler 插件开发指南。当你需要开发、修改或调试 torch_tb_profiler 项目时使用此 Skill。适用于前端 React/TypeScript 组件开发、后端 Python API 开发、性能分析功能增强、UI 组件修改等场景。
---

# PyTorch TensorBoard Profiler 开发指南

本项目是 PyTorch 性能分析器的 TensorBoard 插件，用于可视化 PyTorch 模型的性能数据。

## 项目架构

```
tb_plugin/
├── fe/                          # 前端项目 (React + TypeScript)
│   ├── src/
│   │   ├── app.tsx              # 主应用入口，路由与视图切换
│   │   ├── index.tsx            # 入口文件
│   │   ├── api/                 # API 调用层
│   │   │   ├── generated/       # OpenAPI 自动生成的客户端
│   │   │   └── openapi.yaml     # API 规范定义
│   │   ├── components/          # React 组件
│   │   │   ├── Operator.tsx     # Operator 视图页面
│   │   │   ├── Kernel.tsx       # GPU Kernel 视图页面
│   │   │   ├── Overview.tsx     # 概览页面
│   │   │   ├── MemoryView.tsx   # 内存视图
│   │   │   ├── TraceView.tsx    # Trace 视图
│   │   │   ├── charts/          # 图表组件
│   │   │   │   ├── AntTableChart.tsx  # 通用表格（Kernel 等使用）
│   │   │   │   ├── PieChart.tsx       # 饼图
│   │   │   │   └── ...
│   │   │   └── tables/          # 表格组件
│   │   │       ├── OperationTable.tsx # Operator 表格
│   │   │       ├── common.tsx         # 表格通用逻辑
│   │   │       └── ...
│   │   ├── constants/           # 常量定义
│   │   └── utils/               # 工具函数
│   ├── package.json             # 前端依赖
│   └── webpack.config.js        # Webpack 配置
│
├── torch_tb_profiler/           # 后端 Python 包
│   ├── plugin.py                # TensorBoard 插件主入口，定义路由
│   ├── run.py                   # Run 数据模型
│   ├── profiler/                # 性能分析核心逻辑
│   │   ├── run_generator.py     # 生成表格/图表数据
│   │   ├── op_agg.py            # Operator/Kernel 聚合逻辑
│   │   ├── loader.py            # Trace 加载器
│   │   └── ...
│   ├── io/                      # 存储 I/O（本地、HDFS、GCS 等）
│   └── static/                  # 构建后的前端静态资源
│
├── test/                        # 测试文件
├── examples/                    # 示例脚本
└── setup.py                     # Python 包安装配置
```

## 技术栈

### 前端
- **框架**: React 17+ (函数组件 + Hooks)
- **语言**: TypeScript
- **UI 库**: 
  - Material-UI (@material-ui/core) - 布局、表单控件
  - Ant Design (antd) - 表格 (Table)、分页等
- **状态管理**: React useState/useEffect
- **API 调用**: OpenAPI Generator 生成的客户端

### 后端
- **框架**: TensorBoard 插件 API
- **语言**: Python 3.7+
- **数据处理**: 解析 PyTorch Profiler 的 trace 文件

## 关键文件说明

### 前端核心文件

| 文件路径 | 功能说明 |
|---------|---------|
| `fe/src/app.tsx` | 主应用，根据 view 参数渲染不同页面 |
| `fe/src/components/Operator.tsx` | Operator 视图：饼图 + 算子表格 |
| `fe/src/components/Kernel.tsx` | GPU Kernel 视图：饼图 + kernel 表格 |
| `fe/src/components/tables/OperationTable.tsx` | Operator 数据表格组件 |
| `fe/src/components/charts/AntTableChart.tsx` | 通用表格组件（Kernel 页面使用） |
| `fe/src/components/tables/common.tsx` | 表格列定义和通用逻辑 |
| `fe/src/api/generated/api.ts` | 自动生成的 API 客户端 |

### 后端核心文件

| 文件路径 | 功能说明 |
|---------|---------|
| `torch_tb_profiler/plugin.py` | 插件入口，定义 HTTP 路由 |
| `torch_tb_profiler/run.py` | Run 数据模型和表格数据字段 |
| `torch_tb_profiler/profiler/run_generator.py` | 生成 op/kernel 的表格和饼图数据 |
| `torch_tb_profiler/profiler/op_agg.py` | Operator 和 Kernel 聚合算法 |

## 开发命令

### 前端开发

```bash
# 进入前端目录
cd fe

# 安装依赖
npm install

# 开发模式（热重载）
npm run dev

# 构建生产版本
npm run build

# 生成 API 客户端（当 openapi.yaml 更新后）
npm run generate-api
```

### 后端开发

```bash
# 安装开发模式
pip install -e .

# 运行测试
python -m pytest test/

# 启动 TensorBoard（假设有 profiler 数据）
tensorboard --logdir=./logs
```

## 开发规范

### 前端代码规范

1. **组件结构**: 使用函数组件 + React Hooks
2. **类型定义**: 所有 props 和 state 都要有 TypeScript 类型
3. **样式**: 使用 Material-UI 的 makeStyles 创建样式
4. **表格**: 使用 Ant Design Table 组件，配置 pagination、columns 等
5. **API 调用**: 使用 `api.defaultApi.xxx()` 调用后端接口

### 代码示例

```typescript
// 组件基本结构
import * as React from 'react'
import { Table } from 'antd'
import { makeStyles } from '@material-ui/core/styles'

interface IProps {
  data: YourDataType
}

const useStyles = makeStyles((theme) => ({
  root: { /* 样式 */ }
}))

export const YourComponent: React.FC<IProps> = (props) => {
  const { data } = props
  const classes = useStyles()
  
  // 使用 useMemo 优化计算
  const processedData = React.useMemo(() => {
    return /* 处理数据 */
  }, [data])
  
  return (
    <div className={classes.root}>
      {/* 组件内容 */}
    </div>
  )
}
```

### 表格分页配置

```typescript
// Ant Design Table 分页配置
<Table
  columns={columns}
  dataSource={rows}
  pagination={{
    pageSize,
    pageSizeOptions: ['10', '20', '30', '50', '100'],
    showSizeChanger: true,
    showTotal: (total) => `共 ${total} 条`,
    onShowSizeChange: (current, size) => setPageSize(size)
  }}
/>
```

### 后端代码规范

1. **路由定义**: 在 `plugin.py` 中添加路由
2. **数据处理**: 在 `profiler/` 目录下实现核心逻辑
3. **返回格式**: 使用字典或 dataclass 返回 JSON 数据

## 常见开发任务

### 添加新的表格列

1. 修改 `fe/src/components/tables/common.tsx` 中的列定义
2. 确保后端返回相应的数据字段

### 添加新的 API 接口

1. 在 `fe/src/api/openapi.yaml` 中定义接口规范
2. 运行 `npm run generate-api` 生成客户端代码
3. 在 `torch_tb_profiler/plugin.py` 中实现路由处理函数

### 修改表格分页

相关文件：
- `fe/src/components/tables/OperationTable.tsx` - Operator 表格
- `fe/src/components/charts/AntTableChart.tsx` - Kernel 表格

### 添加数据导出功能

1. 在 `fe/src/utils/` 下创建导出工具函数
2. 在表格组件中添加导出按钮
3. 调用工具函数生成并下载文件

## 已实现的增强功能

### CSV 导出功能

Operator 和 GPU Kernel 表格已支持 CSV 导出功能。

**相关文件:**
- `fe/src/utils/csvExport.ts` - CSV 导出工具函数
- `fe/src/components/tables/OperationTable.tsx` - Operator 表格（包含下载按钮）
- `fe/src/components/charts/AntTableChart.tsx` - Kernel 表格（包含下载按钮）

**使用方法:**
点击表格上方的"下载 CSV"按钮即可导出当前表格的全部数据。

**CSV 导出工具函数:**

```typescript
import { exportToCSV, exportGraphToCSV, generateFilename } from '../../utils/csvExport'

// 导出 Operator 表格数据
const csvColumns = columns.map((col) => ({
  title: String(col.title || ''),
  dataIndex: col.dataIndex as string,
  key: col.key as string
}))
exportToCSV(csvColumns, rows, generateFilename('operator'))

// 导出 Graph 格式数据（Kernel 表格使用）
exportGraphToCSV(graph.columns, graph.rows, generateFilename('kernel'))
```

### 全部显示分页选项

Operator 和 GPU Kernel 表格的分页选项中包含当前数据总行数，选择后可显示全部数据。

**分页配置示例:**

```typescript
pagination={{
  pageSize: actualPageSize,
  pageSizeOptions: ['10', '20', '30', '50', '100', `${rows.length}`],
  showSizeChanger: true,
  showTotal: (total: number) => `共 ${total} 条`,
  onShowSizeChange,
  locale: {
    items_per_page: '条/页'
  }
}}
```

### 时间单位切换功能

Operator 和 GPU Kernel 表格支持时间单位切换（us/ms），所有时间数值保留4位小数。

**相关文件:**
- `fe/src/components/tables/common.tsx` - 时间转换工具函数和类型定义
- `fe/src/components/tables/OperationTable.tsx` - Operator 表格时间单位切换
- `fe/src/components/charts/AntTableChart.tsx` - Kernel 表格时间单位切换

**使用方法:**
在表格上方选择 "us" 或 "ms" 单选按钮，表格中的时间列会自动转换单位并更新列标题。

**核心代码:**

```typescript
// 类型定义
export type TimeUnit = 'us' | 'ms'

// 时间值转换函数
export function convertTimeValue(value: number | undefined, unit: TimeUnit): string {
  if (value === undefined || value === null) {
    return ''
  }
  if (unit === 'ms') {
    return (value / 1000).toFixed(4)
  }
  return value.toFixed(4)
}

// 在列定义中使用 render 函数
{
  dataIndex: 'device_self_duration',
  title: `Device Self Duration (${timeUnit})`,
  render: (value: number) => convertTimeValue(value, timeUnit)
}
```

## 调试技巧

1. **前端调试**: 使用浏览器开发者工具，React DevTools
2. **后端调试**: 使用 print 或 logging 模块输出日志
3. **API 调试**: 在浏览器 Network 面板查看请求响应

## 注意事项

1. 修改前端代码后需要重新构建：`npm run build`
2. 构建产物会输出到 `torch_tb_profiler/static/` 目录
3. 修改 `openapi.yaml` 后需要重新生成 API 客户端
4. 表格数据量大时注意性能优化（使用 useMemo、虚拟滚动等）
