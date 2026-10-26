import argparse
import os
from datasets import load_dataset
from qingxiaotuan import QingxiaotuanTokenizer


def tokenize_function(examples, tokenizer, max_length):
    texts = []
    for text, content in zip(examples.get("text", [None] * len(examples.get("content", []))),
                              examples.get("content", [None] * len(examples.get("text", [])))):
        t = text or content or ""
        texts.append(t)

    tokenizer_output = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
    )

    input_ids = tokenizer_output["input_ids"]
    return {"input_ids": input_ids, "labels": input_ids}


def main():
    parser = argparse.ArgumentParser(description="青小团预训练数据预处理脚本")
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace数据集名称或本地JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="处理后数据输出目录")
    parser.add_argument("--max_length", type=int, default=8192, help="最大序列长度")
    parser.add_argument("--num_proc", type=int, default=8, help="并行处理进程数")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="分词器路径，默认使用qingxiaotuan包内分词器")
    parser.add_argument("--data_files", type=str, default=None, nargs="+", help="本地数据文件列表")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载分词器 QingxiaotuanTokenizer ...")
    if args.tokenizer_path is not None:
        vocab_file = os.path.join(args.tokenizer_path, "qingxiaotuan.tiktoken")
        tokenizer = QingxiaotuanTokenizer(vocab_file=vocab_file)
    else:
        tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")

    print(f"加载数据集: {args.dataset_name}")
    if args.data_files:
        dataset = load_dataset(args.dataset_name, data_files=args.data_files, split="train")
    else:
        dataset = load_dataset(args.dataset_name, split="train")

    print(f"数据集大小: {len(dataset)} 条")
    print(f"开始分词处理，max_length={args.max_length}, num_proc={args.num_proc}")

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="青小团数据分词中",
    )

    output_path = os.path.join(args.output_dir, "tokenized_data")
    print(f"保存处理后数据到: {output_path}")
    tokenized_dataset.save_to_disk(output_path)

    print(f"预处理完成！共处理 {len(tokenized_dataset)} 条样本，保存至 {output_path}")


if __name__ == "__main__":
    main()
