import os
import re

def extract_ops(folder_path):
    """
    从指定文件夹的.log文件中提取所有topsatenCopy()接口调用，并返回去重后的列表

    参数:
        folder_path (str): 包含.log文件的文件夹路径

    返回:
        list: 去重后的接口列表，格式为["topsatenCopy", ...]
    """
    pattern = r'tops\w+\(\)'
    interfaces = set()

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        print(f"Processing file: {filename}")
        if filename.endswith('.log'):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # 查找所有匹配的接口
                    matches = re.findall(pattern,content)
                    interfaces.update(matches)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # 转换为列表并返回
    return sorted(list(interfaces))

# 使用示例
if __name__ == "__main__":
    folder_path = "tops_op_trace"  # 替换为实际的文件夹路径
    interfaces = extract_ops(folder_path)
    print("===> sum: ", len(interfaces))
    print("提取到的去重接口列表:")
    for interface in interfaces:
        print(interface)