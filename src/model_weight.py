import json
import os
import struct

def get_safetensors_metadata(file_path):
    """直接解析 safetensors 文件的头部，极速获取元数据，不加载实际模型权重"""
    with open(file_path, 'rb') as f:
        # 1. 读取前 8 个字节（无符号 64 位小端整数），它表示头部 JSON 的长度
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            return {}
        
        header_size = struct.unpack('<Q', length_bytes)[0]
        
        # 2. 根据长度读取头部 JSON 字符串
        header_json_bytes = f.read(header_size)
        header_dict = json.loads(header_json_bytes.decode('utf-8'))
        
        # safetensors 头部可能包含全局的 '__metadata__'，提取时需将其剔除
        if '__metadata__' in header_dict:
            del header_dict['__metadata__']
            
        return header_dict

def save_weight_info_fast(model_dir, output_file="weight_info.jsonl"):
    try:
        # 找到所有的 safetensors 文件
        safetensors_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        
        if not safetensors_files:
            print(f"在 {model_dir} 目录下没有找到任何 .safetensors 文件。")
            return
            
        print(f"共找到 {len(safetensors_files)} 个 safetensors 文件，开始极速解析...")

        with open(output_file, "w", encoding="utf-8") as out_f:
            for file_name in safetensors_files:
                file_path = os.path.join(model_dir, file_name)
                
                # 瞬间获取文件内所有 tensor 的 shape 和 dtype
                metadata = get_safetensors_metadata(file_path)
                
                for key, details in metadata.items():
                    info = {
                        "key": key,
                        "shape": details.get("shape"),
                        "dtype": details.get("dtype"),
                        "source_file": file_name
                    }
                    out_f.write(json.dumps(info) + "\n")
                    
                print(f"已解析: {file_name} ({len(metadata)} 个张量)")
                
        print(f"\n✅ 权重信息已成功保存到 {output_file}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    model_dir = "/login_home/ckpts/deepseek/DeepSeek-V4-Pro-hygon"  # 替换为你的模型目录
    save_weight_info_fast(model_dir)