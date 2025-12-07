import pandas as pd
from openai import OpenAI
import time
import os
from tqdm import tqdm  # 引入进度条库

# ================= 配置区域 =================

# 2. 阿里云 Qwen 的 OpenAI 兼容 Base URL
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 3. 文件与模型配置
INPUT_FILE = 'dataset/translated_result.csv'
OUTPUT_FILE = 'dataset/scored_result.csv'
BATCH_SIZE = 10
MAX_RETRIES = 10
MODEL_NAME = "qwen-max"
API_KEY_FILE = '.apikey'
# ===========================================

try:
    with open(API_KEY_FILE, 'r', encoding='utf-8') as f:
        API_KEY = f.read().strip()
except FileNotFoundError:
    print(f"错误：找不到 {API_KEY_FILE} 文件")
    exit()

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def get_prompt(batch_data):
    """
    构造 Prompt
    """
    data_str = ""
    for index, row in batch_data.iterrows():
        source = str(row[0]).strip()
        target = str(row[1]).strip() if len(row) > 1 else ""
        data_str += f"{source}, {target}\n"

    prompt = f"""
# Role
你是一位资深的工业与机械工程翻译评估专家。

# Task
请对以下每一行中英文翻译进行质量打分（范围 0-10）。

# Scoring Standards
1. **核对参数（最重要）**：检查型号、规格、数字（如 `YJV`, `3×6mm2`, `E160233`）是否与原文**严格一致**。
   - 如果参数/数字有误，直接打 **0分**。
2. **术语准确性**：
   - **10分**：术语地道，参数完全一致。
   - **8-9分**：意思准确，参数无误，但英文表达略显生硬。
   - **6-7分**：核心意思对，但术语不够专业。
   - **0-5分**：关键术语错误或漏译。

# Output Constraints
- **仅输出数字**（例如：`10`）。
- **不要**加任何文字、标点、前缀（不要写 "Score:"）或解释。
- 必须保持一行对应一个分数，顺序与输入严格一致。
- 输出行数必须与输入行数一致。

# Input Data
{data_str}
"""
    return prompt

def call_qwen_with_retry(batch_data):
    """
    调用模型，包含重试逻辑
    使用 tqdm.write 替代 print 以避免打乱进度条
    """
    prompt_content = get_prompt(batch_data)
    
    for attempt in range(MAX_RETRIES):
        try:
            # 只有在重试的时候才打印 log，正常情况下保持静默，交给进度条
            if attempt > 0:
                tqdm.write(f"  正在重试 API (第 {attempt + 1}/{MAX_RETRIES} 次)...")
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for evaluating translation quality."},
                    {"role": "user", "content": prompt_content}
                ],
                temperature=0.01,
            )

            content = response.choices[0].message.content
            scores = [line.strip() for line in content.split('\n') if line.strip()]
            
            if len(scores) == len(batch_data):
                return scores
            else:
                tqdm.write(f"  [警告] 返回行数不匹配 (预期 {len(batch_data)}, 实际 {len(scores)})。")
        
        except Exception as e:
            tqdm.write(f"  [异常] {e}")
        
        time.sleep(2)
    
    tqdm.write("  [失败] 该批次多次重试失败，标记为 -1")
    return [-1] * len(batch_data)

def main():
    # 1. 读取数据
    print(f"正在读取 {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, header=None)
        if df.shape[1] < 2:
             df[1] = ""
    except FileNotFoundError:
        print("找不到输入文件，请检查路径。")
        return

    total_rows = len(df)
    print(f"共加载 {total_rows} 行数据。开始处理...")

    write_header = not os.path.exists(OUTPUT_FILE)

    # 2. 初始化进度条
    # total=total_rows: 设置总数
    # unit='row': 单位显示为 row
    with tqdm(total=total_rows, unit='row', desc="打分进度") as pbar:
        
        for i in range(0, total_rows, BATCH_SIZE):
            batch = df.iloc[i : i + BATCH_SIZE]
            
            # 调用模型
            scores = call_qwen_with_retry(batch)
            
            # 保存结果
            batch_result = batch.copy()
            batch_result = batch_result.iloc[:, :2] 
            batch_result.columns = ['Source', 'Target']
            batch_result['Score'] = scores
            
            batch_result.to_csv(
                OUTPUT_FILE, 
                mode='a', 
                index=False, 
                header=write_header, 
                encoding='utf-8-sig'
            )
            write_header = False
            
            # 更新进度条
            pbar.update(len(batch))
            
            # 避免过快请求
            time.sleep(0.5)

    print("\n所有处理完成！请查看 scored_result.csv")

if __name__ == "__main__":
    main()