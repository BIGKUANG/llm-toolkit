import re
import pandas as pd
import numpy as np
import sys

def clean_ansi(text):
    """剔除日志中的彩色控制符和 ANSI 转义序列"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_kernel_log(file_path):
    data = []
    
    # 正则表达式说明：
    # 1. 匹配时长行：捕获 ap_dur_time 和 kcore_dur_time 的数字部分
    dur_re = re.compile(r'ap_dur_time:(\d+)ns.*kcore_dur_time:(\d+)ns')
    # 2. 匹配名称行：支持 fun_name 或 func_name，捕获内核函数名
    name_re = re.compile(r'kernel_fun[c]?_name:\s*(\w+)')

    current_ap_ns = None
    current_kcore_ns = None

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 清洗当前行
                line = clean_ansi(line)
                
                # 步骤 A: 查找时长数据
                dur_match = dur_re.search(line)
                if dur_match:
                    current_ap_ns = int(dur_match.group(1))
                    current_kcore_ns = int(dur_match.group(2))
                    continue
                
                # 步骤 B: 查找对应的内核函数名
                name_match = name_re.search(line)
                if name_match and current_ap_ns is not None:
                    kernel_name = name_match.group(1)
                    
                    # 单位转换: ns -> ms (除以 1,000,000)
                    ap_ms = current_ap_ns / 1000000.0
                    kcore_ms = current_kcore_ns / 1000000.0
                    
                    data.append({
                        'kernel_name': kernel_name,
                        'ap_dur_ms': ap_ms,
                        'kcore_dur_ms': kcore_ms,
                        'diff_ms': ap_ms - kcore_ms
                    })
                    
                    # 匹配完成后重置，防止重复关联
                    current_ap_ns = None
                    current_kcore_ns = None

        if not data:
            print("错误：未能在日志中提取到有效数据，请确认关键字是否正确。")
            return

        # 转为 DataFrame 进行聚合运算
        df = pd.DataFrame(data)
        final_stats = []

        # 按内核名称分组统计
        for name, group in df.groupby('kernel_name'):
            # 基础项：调用次数
            row = {'kernel_name': name, 'cycle_count': len(group)}
            
            # 定义需要计算的维度和对应的列名后缀
            metrics = {
                'ap_dur_time': 'ap_dur_ms',
                'kcore_dur_time': 'kcore_dur_ms',
                'diff': 'diff_ms'
            }

            for prefix, col in metrics.items():
                # 基础统计量
                row[f'{prefix}_total_ms'] = group[col].sum()
                row[f'{prefix}_mean_ms']  = group[col].mean()
                row[f'{prefix}_max_ms']   = group[col].max()
                row[f'{prefix}_min_ms']   = group[col].min()
                
                # 百分位统计量
                row[f'{prefix}_25%_ms']   = group[col].quantile(0.25)
                row[f'{prefix}_50%_ms']   = group[col].quantile(0.50)
                row[f'{prefix}_75%_ms']   = group[col].quantile(0.75)
                row[f'{prefix}_99%_ms']   = group[col].quantile(0.99)


            # ---------------- 美化打印部分 ----------------
                print(f"\n{'='*45}")
                print("kernel_name: ", name)
                print(f"📊 性能统计指标: [ {prefix} ]".center(40))
                print(f"{'-'*45}")
                print(f"{'指标名称 (Metrics)':<20} | {'耗时数值 (ms)':>18}")
                print(f"{'-'*45}")
                
                # 打印基础统计量
                print(f"{'总计 (Total)':<20} | {row[f'{prefix}_total_ms']:>18.2f}")
                print(f"{'平均 (Mean)':<20} | {row[f'{prefix}_mean_ms']:>18.2f}")
                print(f"{'最大 (Max)':<20} | {row[f'{prefix}_max_ms']:>18.2f}")
                print(f"{'最小 (Min)':<20} | {row[f'{prefix}_min_ms']:>18.2f}")
                print(f"{'-'*45}")
                
                # 打印百分位统计量
                print(f"{'P25 (25%)':<20} | {row[f'{prefix}_25%_ms']:>18.2f}")
                print(f"{'P50 (50%)':<20} | {row[f'{prefix}_50%_ms']:>18.2f}")
                print(f"{'P75 (75%)':<20} | {row[f'{prefix}_75%_ms']:>18.2f}")
                print(f"{'P99 (99%)':<20} | {row[f'{prefix}_99%_ms']:>18.2f}")
                print(f"{'='*45}\n")

            final_stats.append(row)

        # 整理结果表格
        result_df = pd.DataFrame(final_stats)
        
        # 精度控制：所有数值列保留小数点后 4 位
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.drop('cycle_count')
        result_df[numeric_cols] = result_df[numeric_cols].round(4)

        # 导出 CSV
        output_name = 'kernel_analysis_report.csv'
        result_df.to_csv(output_name, index=False)
        
        print("-" * 30)
        print(f"解析成功！已保存至: {output_name}")
        print(f"统计内核数: {len(result_df)}")
        print("-" * 30)

    except Exception as e:
        print(f"发生异常: {e}")

if __name__ == "__main__":
    # 请确保文件名正确
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'test.log'
    print("===> log_file: ", log_file)
    parse_kernel_log(log_file)