import os
import json
import glob
import re
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ================= 配置区域 =================
# 你微调后保存的模型路径
MODEL_PATH = "saves/llama3-8b-cti-full" 
# 输入报告所在的文件夹路径（假设是txt或md文件）
INPUT_DIR = "./threat_reports"
# 结果保存路径
OUTPUT_FILE = "./extracted_results.jsonl"

# 显卡数量 (您的环境是2张A100)
NUM_GPUS = 2

# 长文本切分配置 (Llama 3.1 虽支持长文，但切分更有利于细节提取)
CHUNK_SIZE = 8000  # 每个片段的字符数粗略估计 (约为 2-3k tokens)
OVERLAP = 500      # 覆盖长度，防止关系被切断

# ================= 系统提示词 =================
SYSTEM_PROMPT = """你是一个专业的网络安全威胁情报分析师。
请阅读给定的威胁情报文本，提取其中提到的关键实体（Entity）以及实体之间的关系（Relation）。
专注于以下实体类型：Attacker(攻击者), Malware(恶意软件), IP, Domain, CVE, Organization, Location, Time。

请严格输出合法的 JSON 列表格式，不要输出任何解释性文字。格式如下：
[
    {"head": "实体1", "type": "实体类型", "relation": "关系描述", "tail": "实体2"},
    ...
]
如果未发现相关信息，输出空列表 []。"""

# ================= 辅助函数类 =================

def load_reports(directory: str) -> List[Dict]:
    """读取目录下所有 .txt 或 .md 文件"""
    files = glob.glob(os.path.join(directory, "*.txt")) + glob.glob(os.path.join(directory, "*.md"))
    reports = []
    print(f"[*] 发现 {len(files)} 份报告，准备处理...")
    for f_path in files:
        with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if content.strip():
                reports.append({"filename": os.path.basename(f_path), "content": content})
    return reports

def split_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """简单的滑动窗口切分文本"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # 尽量在换行符或句号处截断，避免截断单词（简单优化）
        if end < len(text):
            # 向后寻找最近的换行符，限制查找范围
            next_newline = text.find('\n', end, end + 100)
            if next_newline != -1:
                end = next_newline
        
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def clean_json_output(output_text: str):
    """尝试从 LLM 的回复中提取并修复 JSON"""
    try:
        # 1. 尝试直接解析
        return json.loads(output_text)
    except json.JSONDecodeError:
        pass

    try:
        # 2. 提取 markdown 代码块 ```json ... ``` 中的内容
        match = re.search(r"```(?:json)?\s*(.*?)```", output_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
        
        # 3. 提取最外层 [ ... ]
        match = re.search(r"(\[.*\])", output_text, re.DOTALL)
        if match:
             json_str = match.group(1).strip()
             return json.loads(json_str)
             
    except Exception:
        # 解析失败，返回原始内容以便后续人工检查
        return {"error": "parse_failed", "raw_output": output_text}
    
    return []

# ================= 主逻辑 =================

def main():
    # 1. 加载数据
    if not os.path.exists(INPUT_DIR):
        print(f"Error: 输入目录 {INPUT_DIR} 不存在。")
        return
    
    reports = load_reports(INPUT_DIR)
    if not reports:
        print("没有找到报告文件。")
        return

    # 2. 准备 Prompt 数据集
    # 将长报告切分，并构建 inference prompts
    prompts_data = [] # 存储 (filename, chunk_id, ful_prompt)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    print("[*] 正在对文档进行切分与 Prompt 预处理...")
    
    for report in reports:
        chunks = split_text_with_overlap(report['content'], CHUNK_SIZE, OVERLAP)
        for i, chunk in enumerate(chunks):
            # 构建 Llama-3 标准对话格式
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk}
            ]
            # 使用 tokenizer 应用模板，生成最终输入的文本
            text_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts_data.append({
                "filename": report['filename'],
                "chunk_id": i,
                "text_input": text_input
            })

    print(f"[*] 总共生成 {len(prompts_data)} 个文本片段待处理。")

    # 3. 初始化模型 (vLLM)
    # tensor_parallel_size=2 会自动将模型切分到两张卡上
    print(f"[*] 正在加载模型: {MODEL_PATH} 到 {NUM_GPUS} 张 GPU 上...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=NUM_GPUS,
        dtype="bfloat16",         # A100 推荐使用 bf16
        gpu_memory_utilization=0.90, # 显存利用率
        max_model_len=8192,       # 支持的最大上下文长度
        trust_remote_code=True
    )

    # 设定采样参数 (用于实体抽取，temperature 设为 0 以保证确定性)
    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=2048, # 生成的最大长度
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    # 4. 批量推理
    print("[*] 开始批量推理...")
    # 提取所有 prompt 文本列表
    input_prompts = [p["text_input"] for p in prompts_data]
    
    # vLLM 处理推理
    outputs = llm.generate(input_prompts, sampling_params)

    # 5. 解析结果并保存
    print(f"[*] 推理完成，正在保存结果到 {OUTPUT_FILE} ...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, output in enumerate(outputs):
            metadata = prompts_data[i]
            generated_text = output.outputs[0].text
            
            # 清洗和解析 JSON
            extracted_data = clean_json_output(generated_text)
            
            # 构建最终保存对象
            record = {
                "filename": metadata["filename"],
                "chunk_id": metadata["chunk_id"],
                "extracted_info": extracted_data,
                # 可选：如果你如果想保留原始输出进行调试，取消下面注释
                # "raw_llm_output": generated_text 
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[Success] 所有任务已完成！")

if __name__ == "__main__":
    main()
