import json
from safetensors import safe_open


def save_weight_info_to_jsonl(model_dir, output_file="weight_info.jsonl"):
    try:
        # 1. 解析index.json文件
        with open(f"{model_dir}/model.safetensors.index.json", "r") as f:
            index_data = json.load(f)
            weight_map = index_data["weight_map"]
        
        # 2. 按safetensors文件分组key
        file_to_keys = {}
        for key, file_name in weight_map.items():
            if file_name not in file_to_keys:
                file_to_keys[file_name] = []
            file_to_keys[file_name].append(key)
        
        # 3. 写入JSONL文件
        with open(output_file, "w", encoding="utf-8") as out_f:
            for file_name, keys in file_to_keys.items():
                file_path = f"{model_dir}/{file_name}"
                with safe_open(file_path, framework="pt") as f:
                    available_keys = set(f.keys())
                    for key in keys:
                        if key in available_keys:
                            tensor = f.get_tensor(key)  # 先获取张量
                            info = {
                                "key": key,
                                "shape": list(tensor.shape),  # 从张量获取shape
                                "dtype": str(tensor.dtype),  # 从张量获取dtype
                                "source_file": file_name
                            }
                            out_f.write(json.dumps(info) + "\n")
                        else:
                            print(f"Warning: Key '{key}' not found in {file_name}")
        
        print(f"权重信息已成功保存到 {output_file}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    model_dir = "/datasets/deepseek-r1"  # 替换为你的模型目录
    save_weight_info_to_jsonl(model_dir)
