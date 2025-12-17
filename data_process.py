import pandas as pd
import time
import math
import os
import jieba
import collections
import logging
import json
from difflib import SequenceMatcher
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = 'dataset/raw.csv'
OUTPUT_FILE = 'dataset/translated_result.csv' 
LOG_FILE = 'dataset/translation_process.log'
API_KEY_FILE = '.apikey'

BATCH_SIZE = 25          # 批次大小
MAX_RETRIES = 3         # 最大重试次数
MAX_SEARCH_PER_WORD = 20 # TM 搜索深度
SIMILARITY_THRESHOLD = 0.3
DEDUP_THRESHOLD = 0.8
ERROR_MARK = "[[TRANSLATION_FAILED]]" 

SYSTEM_INSTRUCTION = """
你是一个专业的工业自动化与机械制造领域的翻译专家。
任务：将给定的【物料描述】从中文翻译成英文。

核心规则：
1. 【领域适配】使用工业、机械、电子元件的标准英文术语（如：Socket, Cylinder, Relay, Valve）。
2. 【保留参数】绝对禁止修改任何型号、规格、尺寸、数字符号。
   - 例如：'3×6mm2', 'Φ40', 'AC220V', 'YJV', 'S18009-2503' 等必须原样保留。
3. 【品牌处理】知名品牌请使用官方英文名（如：施耐德 -> Schneider, 亚德客 -> AirTAC），未知品牌保留原文或拼音。
4. 【格式一致】保持与原文行数一一对应，不要合并或拆分行。
"""
# ===========================================

# --- 0. 配置日志 ---
logging.getLogger().handlers = []
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# --- 1. 辅助函数 ---
def get_similarity(str1, str2):
    """计算两个字符串的相似度 (0.0 - 1.0)"""
    return SequenceMatcher(None, str1, str2).ratio()

def build_tm_list(current_texts, inverted_index, history_data):
    candidates_map = {} 
    for text in current_texts:
        words = list(jieba.cut(text))
        for word in words:
            if len(word) < 2: continue
            if word in inverted_index:
                reference_indices = inverted_index[word][-MAX_SEARCH_PER_WORD:]
                for idx in reference_indices:
                    hist_src, hist_tgt = history_data[idx]
                    if hist_src in candidates_map: continue
                    sim = get_similarity(text, hist_src)
                    if sim > 0.1: 
                        candidates_map[hist_src] = (hist_tgt, sim)
    sorted_candidates = sorted(candidates_map.items(), key=lambda x: x[1][1], reverse=True)
    final_tm_list = []
    for src, (tgt, score) in sorted_candidates:
        if len(final_tm_list) >= 50: break
        is_redundant = False
        for selected in final_tm_list:
            if get_similarity(src, selected['source']) > DEDUP_THRESHOLD:
                is_redundant = True
                break 
        if not is_redundant:
            final_tm_list.append({"source": src, "target": tgt})
    return final_tm_list

def verify_with_qwen_max(source_list, candidate_list):
    if len(source_list) != len(candidate_list): return False
    check_prompt = "请校验以下【原文列表】与【译文列表】是否内容对应，且行数一致。\n"
    check_prompt += "如果完全对应，请仅回复 YES。如果不对应或有错位，请回复 NO。\n\n"
    check_prompt += "【原文列表】:\n" + "\n".join(source_list) + "\n\n"
    check_prompt += "【译文列表】:\n" + "\n".join(candidate_list)
    try:
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "user", "content": check_prompt}]
        )
        return "YES" in completion.choices[0].message.content.strip().upper()
    except Exception as e:
        logger.error(f"Verification API Error: {e}")
        return False

# --- 2. 初始化与加载 ---
try:
    api_key = open(API_KEY_FILE).read().strip()
except FileNotFoundError:
    print(f"错误：找不到 {API_KEY_FILE} 文件")
    exit()

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

try:
    # 加载原始数据（保留原始顺序的关键）
    df_raw = pd.read_csv(INPUT_FILE, encoding='UTF-8', header=None, names=['src'])
    # 确保列名是字符串，防止只有数字的描述被当成 int
    all_source_list = df_raw['src'].astype(str).tolist()
    print(f"原始数据加载成功: {len(all_source_list)} 条")
except Exception as e:
    print(f"读取原始文件失败: {e}")
    exit()

history_data = [] 
inverted_index = collections.defaultdict(list) 
completed_map = {} 

if os.path.exists(OUTPUT_FILE):
    print("检测到历史文件，正在分析进度...")
    try:
        df_done = pd.read_csv(OUTPUT_FILE, header=None, names=['src', 'tgt'], encoding='UTF-8')
        # 倒序遍历：这样如果有重复的 key，前面的（旧的/失败的）会被后面的（新的/成功的）覆盖
        # 这确保了 completed_map 里存的一定是最新一次的状态
        for idx, row in df_done.iterrows():
            src = str(row['src'])
            tgt = str(row['tgt'])
            
            completed_map[src] = tgt
            
            if tgt != ERROR_MARK and tgt.strip() != "":
                current_hist_idx = len(history_data)
                history_data.append((src, tgt))
                for w in jieba.cut(src):
                    if len(w) >= 2:
                        inverted_index[w].append(current_hist_idx)
        print(f"--- 已加载历史记忆: {len(history_data)} 条 ---")
    except Exception as e:
        print(f"警告：读取历史文件失败 ({e})，将重新开始。")

tasks_to_process = [] 
for idx, src in enumerate(all_source_list):
    if src not in completed_map or completed_map[src] == ERROR_MARK:
        tasks_to_process.append(src)

print(f"--- 总任务数: {len(all_source_list)} | 待处理: {len(tasks_to_process)} ---")

if not os.path.exists(OUTPUT_FILE):
    pass
    # pd.DataFrame(columns=['物料描述', '英文描述']).to_csv(OUTPUT_FILE, index=False, encoding='UTF-8')

# --- 3. 批处理循环 ---
with tqdm(total=len(tasks_to_process), unit="row", desc="Translating") as pbar:
    for i in range(0, len(tasks_to_process), BATCH_SIZE):
        batch_source = tasks_to_process[i : i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        current_tm_list = build_tm_list(batch_source, inverted_index, history_data)
        translation_options = {
            "source_lang": "zh", "target_lang": "en",
            "tm_list": current_tm_list, "domains": SYSTEM_INSTRUCTION
        }
        input_text = "\n".join(batch_source)
        batch_results = []
        success = False

        # --- LOG: 记录输入信息 ---
        log_entry_input = {
            "batch_index": i,
            "tm_count": len(current_tm_list),
            "tm_content": current_tm_list, # 记录这一批次用到的具体参考
            "input_text": input_text
        }
        logger.info(f"====== BATCH {current_batch_num} INPUT ======\n{json.dumps(log_entry_input, ensure_ascii=False)}")

        for attempt in range(MAX_RETRIES):
            try:
                messages = [
                    {"role": "user", "content": input_text}
                ]
                completion = client.chat.completions.create(
                    model="qwen-mt-plus",
                    messages=messages,
                    extra_body={"translation_options": translation_options}
                )
                response_text = completion.choices[0].message.content
                output_list = response_text.strip().split('\n')
                logger.info(f"====== OUTPUT (Attempt {attempt+1}) ======\n{response_text[:200]}")

                if len(output_list) == len(batch_source):
                    batch_results = output_list
                    success = True
                    break
                elif len(output_list) > len(batch_source):
                    candidate_fix = output_list[-len(batch_source):]
                    tqdm.write(f"  [Batch {current_batch_num}] 长度修复校验...")
                    if verify_with_qwen_max(batch_source, candidate_fix):
                        batch_results = candidate_fix
                        success = True
                        logger.info("Heuristic fix verified.")
                        break
                    else:
                        tqdm.write(f"  [Batch {current_batch_num}] 截取校验失败，重试中...")
                else:
                    tqdm.write(f"  [Batch {current_batch_num}] 行数不足 ({len(output_list)}/{len(batch_source)})，重试 {attempt+1}...")

            except Exception as e:
                logger.error(f"Batch Error: {e}")
                tqdm.write(f"  [Error] {e} - Retrying...")
                time.sleep(1)

        if not success:
            logger.error(f"Batch {current_batch_num} FAILED.")
            tqdm.write(f"  !!! 批次失败，标记为 {ERROR_MARK}")
            batch_results = [ERROR_MARK] * len(batch_source)
        
        # 实时保存（Append模式，确保进度不丢失）
        try:
            df_batch = pd.DataFrame({'src': batch_source, 'tgt': batch_results})
            df_batch.to_csv(OUTPUT_FILE, mode='a', header=None, index=False, encoding='UTF-8')
            
            if success:
                start_hist_idx = len(history_data)
                for idx, (src, tgt) in enumerate(zip(batch_source, batch_results)):
                    if tgt != ERROR_MARK:
                        history_data.append((src, tgt))
                        abs_idx = start_hist_idx + idx
                        for w in jieba.cut(src):
                            if len(w) >= 2: inverted_index[w].append(abs_idx)
        except Exception as e:
            logger.error(f"Save Error: {e}")

        pbar.update(len(batch_source))

# --- 4. 最终重组 (Post-Processing) ---
print("\n" + "="*30)
print("正在执行最终重组 (Re-ordering)...")
try:
    # 1. 读取包含所有追加记录的进度文件
    df_progress = pd.read_csv(OUTPUT_FILE, header=None, names=['src', 'tgt'], encoding='UTF-8')
    
    # 2. 创建映射字典 (Dictionary)
    # 转换为字符串防止数字类型的 key 匹配失败
    progress_map = dict(zip(df_progress['src'].astype(str), df_progress['tgt'].astype(str)))
    
    # 3. 准备最终有序的 DataFrame
    df_final = df_raw.copy() # 此时只有 'src' 列
    
    # 4. 映射翻译结果 [已修复]
    # 逻辑：新建 'tgt' 列 = 根据 'src' 列的内容去 progress_map 查表
    df_final['tgt'] = df_final['src'].astype(str).map(progress_map)
    
    # 5. 填补可能的遗漏
    df_final['tgt'] = df_final['tgt'].fillna(ERROR_MARK)
    
    # 6. 覆盖保存
    # 如果你希望最终文件包含表头，可以将 header=None 改为 header=['物料描述', '英文描述']
    df_final.to_csv(OUTPUT_FILE, index=False, header=None, encoding='UTF-8')
    
    print(f"重组完成！文件已恢复原始顺序。")
    print(f"最终结果保存在: {OUTPUT_FILE}")
    
    # 简单统计 [已修复列名引用]
    fail_count = len(df_final[df_final['tgt'] == ERROR_MARK])
    
    if fail_count > 0:
        print(f"注意：仍有 {fail_count} 条数据翻译失败，标记为 {ERROR_MARK}")
        print("请再次运行脚本以重试这些行。")
    else:
        print("完美！所有数据已翻译完成且顺序正确。")

except Exception as e:
    logger.error(f"Re-ordering Failed: {e}")
    # 打印详细错误堆栈，方便调试
    import traceback
    traceback.print_exc()
    print(f"!!! 最终重组失败: {e}")

print("="*30)