import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# =================配置区域=================
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  # 或者是你本地权重的路径
DATA_PATH = "dataset_train.jsonl"              # 上一步生成的训练数据
OUTPUT_DIR = "./llama3_1_full_finetune_output"
MAX_SEQ_LENGTH = 2048                          # 威胁情报文本较长，建议2048或4096
# ==========================================

def format_chat_template(row, tokenizer):
    """
    将 mrmoor 数据集的 instruction/input/output 格式
    转换为 Llama 3 官方的 Chat Template 格式
    """
    # 构造 Llama 3 的对话列表
    messages = [
        {"role": "system", "content": row['instruction']},
        {"role": "user", "content": row['input']},
        {"role": "assistant", "content": row['output']}
    ]
    
    # 使用 tokenizer 自动应用模板，并在末尾添加生成结束符
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    return {"text": text}

def main():
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Llama 3 需要将 pad 设为 eos
    tokenizer.padding_side = "right" # 训练时通常设为 right

    # 2. 加载并处理数据集
    print("正在加载数据集...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    print("正在应用 Chat Template...")
    # 将 JSONL 转换为模型可读的 prompt 格式
    dataset = dataset.map(lambda row: format_chat_template(row, tokenizer))

    # Tokenizer 处理 (Tokenization)
    def process_func(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Label 即为 input_ids，但在 loss 计算时会自动忽略 padding 部分
        labels = model_inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100 # -100 会被 PyTorch 忽略计算 Loss
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        process_func,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=8
    )

    # 3. 加载模型 (全量加载，不量化)
    print("正在加载模型 (这可能需要大量内存)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,    # Llama 3 必须使用 bf16 以保持数值稳定性
        use_cache=False,               # 训练时必须关闭 cache
        attn_implementation="flash_attention_2" # 强烈建议开启 Flash Attention
    )
    
    # 开启梯度检查点 (Gradient Checkpointing) 以节省大量显存
    model.gradient_checkpointing_enable()

    # 4. 设置训练参数 (Training Arguments)
    # 注意：全量微调通常需要 FSDP 或 DeepSpeed
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        
        # --- 核心训练参数 ---
        num_train_epochs=3,              # 训练轮数
        per_device_train_batch_size=2,   # 显存不够就调小这个，或增大 gradient_accumulation_steps
        gradient_accumulation_steps=8,   # 累计梯度，模拟大 Batch Size
        learning_rate=2e-5,              # 全量微调学习率要低 (LoRA通常2e-4，这里建议 1e-5 ~ 2e-5)
        weight_decay=0.01,
        
        # --- 硬件优化 ---
        bf16=True,                       # 必须开启 BF16
        fp16=False,                      # Llama 3 不推荐 FP16
        gradient_checkpointing=True,     # 显存优化关键
        
        # --- FSDP 配置 (如果是多卡环境) ---
        # 如果你使用 accelerate config 或 torchrun 启动，这里可以自动适配
        # 也可以在这里硬编码 fsdp="full_shard auto_wrap"
        
        # --- 日志与保存 ---
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="tensorboard",
        dataloader_num_workers=4,
    )

    # 5. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    # 6. 开始训练
    print("开始全量微调...")
    trainer.train()

    # 7. 保存最终模型
    print(f"训练完成，正在保存模型至 {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
