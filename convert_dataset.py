import json
import random
from typing import List, Dict

# ==========================================
# 1. 这是目标指令格式 (Prompt Template)
# ==========================================
SYSTEM_PROMPT = (
    "你是一位网络安全情报专家。请分析下方的威胁情报文本，"
    "提取其中的关键实体（如 Attacker, Malware, CVE 等）以及它们之间的关系。"
    "请严格按照 JSON 格式输出结果。"
)

# ==========================================
# 2. 模拟数据生成器 (用于演示)
# 如果你已经有真实数据文件，可以跳过此函数
# ==========================================
def generate_mock_mrmoor_data(num_samples=5):
    """
    模拟 mrmoor 数据集的原始结构。
    通常包含：文本、实体列表（带ID）、关系列表（引用ID）。
    """
    mock_data = []
    sample_texts = [
        "APT28 uses Zebrocy malware to target government sectors.",
        "Lazarus Group exploits CVE-2021-44228 against financial institutions.",
        "BlackCat ransomware encrypts files using AES algorithm."
    ]
    
    for i in range(num_samples):
        text = random.choice(sample_texts)
        # 这里模拟原始数据的结构
        item = {
            "text": text,
            "entities": [
                # 假设源数据有 ID, span, label, text
                {"id": "T1", "label": "Attacker", "text": "APT28", "start": 0, "end": 5},
                {"id": "T2", "label": "Malware", "text": "Zebrocy", "start": 11, "end": 18}
            ],
            "relations": [
                # 关系通常引用实体的 ID
                {"head_id": "T1", "tail_id": "T2", "type": "Uses"}
            ]
        }
        mock_data.append(item)
    return mock_data

# ==========================================
# 3. 核心转换逻辑
# ==========================================
def convert_to_instruction_format(raw_data: List[Dict]) -> List[Dict]:
    """
    将原始 CTI 数据转换为 LLM 微调所需的 Instruction 格式 (Alpaca style)。
    格式: {"instruction": ..., "input": ..., "output": ...}
    """
    formatted_dataset = []

    for item in raw_data:
        original_text = item.get("text", "")
        entities = item.get("entities", [])
        relations = item.get("relations", [])

        # --- 步骤 A: 构建实体查找表 (ID -> Name) ---
        # 很多原始数据集的关系只记录了 ID，我们需要把它翻译成具体的文本名字
        entity_id_map = {e["id"]: e["text"] for e in entities}

        # --- 步骤 B: 构建期望的输出 (Gold Standard Output) ---
        # 这是我们希望 LLM 学会吐出的 JSON 结构
        output_structure = {
            "entities": [],
            "relations": []
        }

        # 1. 填充实体
        for e in entities:
            output_structure["entities"].append({
                "name": e["text"],
                "type": e["label"]
            })

        # 2. 填充关系 (需要把 T1, T2 这种 ID 换成文本)
        for r in relations:
            head_text = entity_id_map.get(r["head_id"])
            tail_text = entity_id_map.get(r["tail_id"])
            
            # 只有当头尾实体都能找到对应文本时才添加
            if head_text and tail_text:
                output_structure["relations"].append({
                    "head": head_text,
                    "relation": r["type"],
                    "tail": tail_text
                })

        # --- 步骤 C: 组装成微调数据条目 ---
        # 将 output_structure 转为字符串，因为 LLM 训练是一个生成文本的过程
        output_str = json.dumps(output_structure, ensure_ascii=False)

        entry = {
            "instruction": SYSTEM_PROMPT,
            "input": original_text,
            "output": output_str
        }

        formatted_dataset.append(entry)

    return formatted_dataset

# ==========================================
# 4. 主执行流程
# ==========================================
def main():
    # settings
    INPUT_FILE = "raw_mrmoor_data.json" # 假设这是你的源文件
    OUTPUT_TRAIN_FILE = "dataset_train.jsonl"
    OUTPUT_VAL_FILE = "dataset_val.jsonl"
    
    print("1. 加载/生成数据...")
    # 注意：实际使用时，请用 json.load(open(INPUT_FILE)) 替换下面这行
    raw_data = generate_mock_mrmoor_data(100) 
    
    print(f"   共获取 {len(raw_data)} 条原始数据。")

    print("2. 格式转换中...")
    llm_dataset = convert_to_instruction_format(raw_data)

    # 3. 划分训练集和验证集 (比如 90% 训练, 10% 验证)
    random.shuffle(llm_dataset)
    split_idx = int(len(llm_dataset) * 0.9)
    train_data = llm_dataset[:split_idx]
    val_data = llm_dataset[split_idx:]

    print(f"3. 写入文件: 训练集 ({len(train_data)}), 验证集 ({len(val_data)})")

    # 写入 JSONL (JSON Lines) 格式，这是微调最常用的格式
    with open(OUTPUT_TRAIN_FILE, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(OUTPUT_VAL_FILE, 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n转换完成！")
    print("样例预览 (第一条训练数据):")
    print(json.dumps(train_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
