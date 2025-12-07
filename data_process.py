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
DEDUP_THRESHOLD = 0.8   # TM 去重阈值

# 定义错误标记 (当重试失败时写入此标记，以便下次识别并重试)
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
    """构建翻译记忆 (TM) 列表 - 优化版 (含去重与排序)"""
    candidates_map = {} 
    
    for text in current_texts:
        words = list(jieba.cut(text))
        for word in words:
            if len(word) < 2: continue
            
            if word in inverted_index:
                # 获取该词对应的历史索引 (取最近的 N 条)
                reference_indices = inverted_index[word][-MAX_SEARCH_PER_WORD:]
                for idx in reference_indices:
                    hist_src, hist_tgt = history_data[idx]
                    
                    if hist_src in candidates_map: continue

                    sim = get_similarity(text, hist_src)
                    
                    # 只有相似度 > 0.1 才入围
                    if sim > 0.1: 
                        candidates_map[hist_src] = (hist_tgt, sim)

    # 按相似度降序
    sorted_candidates = sorted(candidates_map.items(), key=lambda x: x[1][1], reverse=True)

    final_tm_list = []
    
    for src, (tgt, score) in sorted_candidates:
        if len(final_tm_list) >= 50: break
            
        is_redundant = False
        # 内部去重
        for selected in final_tm_list:
            if get_similarity(src, selected['source']) > DEDUP_THRESHOLD:
                is_redundant = True
                break 
        
        if not is_redundant:
            final_tm_list.append({"source": src, "target": tgt})

    return final_tm_list

def verify_with_qwen_max(source_list, candidate_list):
    """使用 qwen-max 校验行数对应关系"""
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
        answer = completion.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception as e:
        logger.error(f"Verification API Error: {e}")
        return False

# --- 2. 初始化与数据加载 (核心修改部分) ---

try:
    api_key = open(API_KEY_FILE).read().strip()
except FileNotFoundError:
    print(f"错误：找不到 {API_KEY_FILE} 文件")
    exit()

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2.1 读取原始数据
try:
    df_raw = pd.read_csv(INPUT_FILE, encoding='UTF-8')
    all_source_list = df_raw['物料描述'].astype(str).tolist()
    print(f"原始数据加载成功: {len(all_source_list)} 条")
except Exception as e:
    print(f"读取原始文件失败: {e}")
    exit()

# 2.2 初始化 TM 数据结构
history_data = [] 
inverted_index = collections.defaultdict(list) 

# 2.3 智能加载历史进度 (处理断点与错误重试)
completed_map = {} # 用于快速查找已完成的任务 {source: target}

if os.path.exists(OUTPUT_FILE):
    print("检测到历史文件，正在分析进度...")
    try:
        # 读取历史文件
        df_done = pd.read_csv(OUTPUT_FILE)
        
        # 遍历历史记录
        for idx, row in df_done.iterrows():
            src = str(row['物料描述'])
            tgt = str(row['英文描述'])
            
            # 记录到 map 中，如果有重复，后面会覆盖前面（保留最新的状态）
            completed_map[src] = tgt
            
            # 【关键】只有当翻译成功且不是错误标记时，才加入 TM (记忆库)
            if tgt != ERROR_MARK and tgt.strip() != "":
                current_hist_idx = len(history_data)
                history_data.append((src, tgt))
                for w in jieba.cut(src):
                    if len(w) >= 2:
                        inverted_index[w].append(current_hist_idx)
                        
        print(f"--- 已加载历史记忆: {len(history_data)} 条 ---")
        
    except Exception as e:
        print(f"警告：读取历史文件失败 ({e})，将重新开始。")

# 2.4 构建待处理任务列表 (Task List)
tasks_to_process = [] # 存放 (原始索引, 物料描述)

for idx, src in enumerate(all_source_list):
    # 判据：如果不在 completed_map 中，或者在但标记为 ERROR_MARK，则需要处理
    if src not in completed_map or completed_map[src] == ERROR_MARK:
        tasks_to_process.append(src)

print(f"--- 总任务数: {len(all_source_list)} | 已完成: {len(all_source_list) - len(tasks_to_process)} | 待处理: {len(tasks_to_process)} ---")

# 如果没有输出文件，先创建表头
if not os.path.exists(OUTPUT_FILE):
    pd.DataFrame(columns=['物料描述', '英文描述']).to_csv(OUTPUT_FILE, index=False, encoding='UTF-8')


# --- 3. 批处理循环 (针对待处理列表) ---

# 注意：这里我们遍历的是 tasks_to_process，而不是 all_source_list
with tqdm(total=len(tasks_to_process), unit="row", desc="Translating") as pbar:
    
    for i in range(0, len(tasks_to_process), BATCH_SIZE):
        batch_source = tasks_to_process[i : i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        # A. 构建 TM
        current_tm_list = build_tm_list(batch_source, inverted_index, history_data)
        
        translation_options = {
            "source_lang": "zh",
            "target_lang": "en",
            "tm_list": current_tm_list,
            "domains": SYSTEM_INSTRUCTION
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

        # B. 调用 API
        for attempt in range(MAX_RETRIES):
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_INSTRUCTION}, # 注入 System Prompt
                    {"role": "user", "content": input_text}
                ]
                
                completion = client.chat.completions.create(
                    model="qwen-mt-plus",
                    messages=messages,
                    extra_body={"translation_options": translation_options}
                )
                
                response_text = completion.choices[0].message.content
                output_list = response_text.strip().split('\n')
                
                # --- LOG ---
                logger.info(f"====== BATCH OUTPUT (Attempt {attempt+1}) ======\n{response_text[:200]}...")

                # 校验逻辑
                if len(output_list) == len(batch_source):
                    batch_results = output_list
                    success = True
                    break
                
                elif len(output_list) > len(batch_source):
                    # 尝试修复：截取后 N 行
                    candidate_fix = output_list[-len(batch_source):]
                    tqdm.write(f"  [Batch {current_batch_num}] 行数冗余，尝试智能截取...")
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

        # C. 错误处理与保存 (核心修改)
        if not success:
            # 【关键】如果重试耗尽仍失败，填入错误标记
            logger.error(f"Batch {current_batch_num} FAILED after {MAX_RETRIES} retries.")
            tqdm.write(f"  !!! 本批次处理失败，已标记为 {ERROR_MARK} !!!")
            batch_results = [ERROR_MARK] * len(batch_source)
        
        # D. 保存结果 (Append 模式)
        try:
            # 即使失败也写入，保证进度指针向前移动
            df_batch = pd.DataFrame({
                '物料描述': batch_source,
                '英文描述': batch_results
            })
            # mode='a' 追加写入
            df_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='UTF-8')
            
            # E. 只有成功的才更新内存 TM
            if success:
                start_hist_idx = len(history_data)
                for idx, (src, tgt) in enumerate(zip(batch_source, batch_results)):
                    # 双重保险：不要把错误标记写进历史
                    if tgt != ERROR_MARK:
                        history_data.append((src, tgt))
                        abs_idx = start_hist_idx + idx
                        for w in jieba.cut(src):
                            if len(w) >= 2:
                                inverted_index[w].append(abs_idx)
                                
        except Exception as e:
            logger.error(f"Save File Error: {e}")
            tqdm.write(f"  !!! 致命错误：无法写入文件 {e}")

        pbar.update(len(batch_source))

print("\n" + "="*30)
print("处理结束！")
print(f"请检查 {OUTPUT_FILE}")
print(f"如果有行显示 {ERROR_MARK}，请重新运行脚本，程序将自动识别并重试这些行。")