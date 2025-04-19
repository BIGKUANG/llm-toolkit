# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import argparse
from vllm import LLM, SamplingParams
import os
import time

benchmark_respond = ""
dump_dir = "/tmp/vllm_dump"



def check_response(outputs):
    for index, output in enumerate(outputs):
        # Check if the output is empty
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"request_id: {output.request_id} promopt: {prompt} Generated text: {generated_text!r}")
        if generated_text != benchmark_respond:
            return False
        

def init_model(args):
    llm = LLM(model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=4,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            quantization=args.quantization,
            dtype=args.dtype,
            enforce_eager=args.enforce_eager,
            enable_expert_parallel=args.enable_expert_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=0)
    return llm


def infer(args, prompts, batch_size=16, num_iter=1):
    # Initialize the LLM model
    llm = init_model(args)

    # Create a sampling params object.
    # sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, ignore_eos=False, seed=0)
    sampling_params = SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=16384, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None)
   
    dp = 8
    num_replicas = (batch_size//dp) 
    prompts = prompts * num_replicas
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    for i in range(num_iter):
        print("===> Iteration: ", i)
        outputs = llm.generate(prompts, sampling_params)

        assert len(outputs) == batch_size, "output batch size mismatch"

        status = check_response(outputs)
        if status is False:
            ## interrupt
            assert 0, "find mismatch in the output"
        
        ## delete dump data
        if num_iter > 1:
            os.path.rmdir(dump_dir)
        time.sleep(1)

 
if __name__ == '__main__':
    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_model_len', type=int, default=2048, help='Maximum model context length')
    parser.add_argument('--quantization', type=str, default='moe_wna16_gcu', help='Quantization method')
    parser.add_argument('--max_num_seqs', type=int, default=1, help='Maximum number of sequences')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for model')
    parser.add_argument('--model', type=str, default='/home/pretrained_models/DeepSeek-R1-awq/', help='Model path')
    parser.add_argument('--enforce_eager', action='store_true', default=False, help='Enable eager execution mode')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization ratio')
    parser.add_argument('--enable-expert-parallel', action='store_true', help='Use expert parallelism instead of tensor parallelism for MoE layers.')
    args = parser.parse_args()

    prompts = [
        "<｜begin▁of▁sentence｜><｜User｜>Hi I have a JavaRDD data named onsite attributed data which consists fields of query and timestamp. I have another JavaRDD consists of top frequent queries. For each record in onsite attributed data, if the query is present in top frequent queries dataset and if it's first time the system would count this as a cache miss, send a request to retrieve such query and cache in the memory for 12 hours. Then next time the same query comes, it would be a cache hit. Show me a sample Spark job to calculate the hit rate for onsite attributed data.<｜Assistant｜>",
    ]

    # Run the inference
    infer(args, prompts, batch_size=16, num_iter=1)
 
