import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。",
        "如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。如果你是刚开始接触分布式训练，建议先看 torch.nn.parallel.DistributedDataParallel，它是封装好的 dist 高级类。而底层的 dist.all_reduce 等函数，通常在你需要手写复杂的分布式算法时才会用到。在 PyTorch 中，torch.distributed（通常简写为 dist）是用于分布式训练的核心模块。它的主要作用是让你的模型能够在多张 GPU 显卡（甚至多台服务器）上同时运行，并同步它们之间的数据。如果不使用 dist，你的程序只能在一个进程里运行；有了它，你可以启动多个进程，每个进程控制一块 GPU。",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
