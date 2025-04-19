from vllm import SamplingParams, LLM
from vllm.assets.video import VideoAsset


def init_qwen2_5_vl_model():
    """初始化 Qwen2.5-VL 模型"""
    model_name = "/home/dakuang.shen/vfarm-share/big-model/Qwen2.5-VL-7B-Instruct"
    
    llm = LLM(
        model=model_name,
        max_model_len=65536,
        max_num_seqs=5,
        tensor_parallel_size=4,  # 设置使用4卡进行张量并行
        dtype="float16", 
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 720,
            "fps": 5,
        },
    )
    return llm

def prepare_qwen2_5_vl_input(question: str, modality: str):
    """准备 Qwen2.5-VL 模型的输入数据"""
    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return prompt, stop_token_ids

def test_qwen2_5_vl(examples):
    # 初始化模型
    llm = init_qwen2_5_vl_model()
    # 创建报告文件
    with open("report.txt", "w", encoding="utf-8") as report_file:
        # 遍历每个示例
        for example in examples:
            # 获取问题和视频路径
            question = example["question"]
            video_path = example["video_path"]
            
            # 加载视频
            video = VideoAsset(video_path, num_frames=60).np_ndarrays
            
            # 设置模态
            modality = "video"
            

            # llm, prompt, stop_token_ids = run_qwen2_5_vl(question, modality)

            # 数据处理
            prompt, stop_token_ids = prepare_qwen2_5_vl_input(question, modality)

            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=0.2,
                max_tokens=1024,
                stop_token_ids=stop_token_ids
            )
            
            # 准备输入
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {
                    modality: video
                }
            }
            
            # 运行推理
            outputs = llm.generate(inputs, sampling_params=sampling_params)
            
            # 获取模型响应
            response = outputs[0].outputs[0].text
            
            # 写入报告
            report_file.write(f"问题: {question}\n")
            report_file.write(f"视频路径: {video_path}\n")
            report_file.write(f"模型响应: {response}\n")
            report_file.write("-" * 50 + "\n")
            
            # 打印当前处理进度
            print(f"已处理: {video_path}")
            print(f"响应: {response}\n")

if __name__ == "__main__":

    base_path = "/home/dakuang.shen/vfarm-share/framwork/vllm/examples/offline_inference/data"
    examples = [
        {"question": "视频中发生了什么？对我的车辆有没有威胁", "video_path": f"{base_path}/Sentry1.mp4"},
        {"question": "检测视频中是否存在汽车开门撞到了另外一辆车", "video_path": f"{base_path}/Sentry2.mp4"},
        {"question": "检测视频中是否存在汽车开门撞到了另外一辆车", "video_path": f"{base_path}/Sentry3.mp4"},
        {"question": "视频中发生了什么？对我的车辆有没有威胁", "video_path": f"{base_path}/Sentry4.mp4"},
        {"question": "视频中发生了什么？对我的车辆有没有威胁", "video_path": f"{base_path}/Sentry5.mp4"},
        {"question": "视频中发生了什么？对我的车辆有没有威胁", "video_path": f"{base_path}/Sentry6.mp4"},
        {"question": "视频中发生了什么？对我的车辆有没有威胁", "video_path": f"{base_path}/Sentry7.mp4"},
        {"question": "视频中发生了什么？对我的车辆有没有威胁", "video_path": f"{base_path}/Sentry8.mp4"},
        {"question": "视频中发生了什么？对我的车辆有没有威胁", "video_path": f"{base_path}/Sentry9.mp4"}
    ]
    test_qwen2_5_vl(examples)