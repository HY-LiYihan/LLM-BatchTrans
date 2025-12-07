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
from tqdm import tqdm  # 引入进度条库

# ================= 配置区域 =================
INPUT_FILE = 'dataset/raw.csv'
OUTPUT_FILE = 'dataset/translated_result.csv' 
LOG_FILE = 'dataset/translation_process.log'   # 新增：日志文件路径
API_KEY_FILE = '.apikey'
BATCH_SIZE = 25
MAX_RETRIES = 3
# 为了避免倒排索引过大导致卡顿，限制每个词检索的历史记录数量
MAX_SEARCH_PER_WORD = 20 
SIMILARITY_THRESHOLD = 0.3
DEDUP_THRESHOLD = 0.8      # 内部去重阈值：如果两条历史记录之间的相似度超过此值，只保留排名靠前的那条
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

# --- 0. 配置日志 (新增) ---
# 清空之前的 handlers 避免重复打印
logging.getLogger().handlers = []
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8' # 确保中文正常记录
)
logger = logging.getLogger(__name__)

# --- 1. 辅助函数 ---

def get_similarity(str1, str2):
    """计算两个字符串的相似度 (0.0 - 1.0)"""
    return SequenceMatcher(None, str1, str2).ratio()

def build_tm_list(current_texts, inverted_index, history_data):
    """
    构建翻译记忆 (TM) 列表 - 优化版
    逻辑：全文检索 -> 按相似度排序 -> 内部去重 -> 截取 Top N
    """
    # 1. 【海选】收集所有潜在候选
    # 格式: { history_source: (history_target, similarity_score) }
    candidates_map = {} 

    for text in current_texts:
        words = list(jieba.cut(text))
        for word in words:
            if len(word) < 2:
                continue
            
            if word in inverted_index:
                # 获取该词对应的历史索引（取最近的 N 条，保证时效性）
                reference_indices = inverted_index[word][-MAX_SEARCH_PER_WORD:]
                
                for idx in reference_indices:
                    hist_src, hist_tgt = history_data[idx]
                    
                    # 如果已经计算过该句子的相似度，跳过
                    if hist_src in candidates_map:
                        continue

                    # 计算与当前输入文本的相似度
                    sim = get_similarity(text, hist_src)
                    
                    # 初步筛选：只有相似度达到一定门槛才进入候选池 (例如 0.1)
                    # 避免完全不相关的句子干扰排序
                    if sim > 0.1: 
                        candidates_map[hist_src] = (hist_tgt, sim)

    # 2. 【排序】按相似度从高到低排序
    # sorted_candidates 是一个 list: [(src, (tgt, score)), ...]
    sorted_candidates = sorted(candidates_map.items(), key=lambda x: x[1][1], reverse=True)

    # 3. 【去重与截取】
    final_tm_list = []


    for src, (tgt, score) in sorted_candidates:
        if len(final_tm_list) >= 50: # 最终数量限制
            break
            
        is_redundant = False
        # 检查当前候选与已选入的 candidates 是否过于相似
        for selected in final_tm_list:
            existing_src = selected['source']
            # 计算两条历史记录之间的相似度
            inter_sim = get_similarity(src, existing_src)
            if inter_sim > DEDUP_THRESHOLD:
                is_redundant = True
                break # 发现重复，丢弃当前候选
        
        if not is_redundant:
            final_tm_list.append({
                "source": src, 
                "target": tgt
                # "score": score # 调试时可开启
            })

    return final_tm_list


def verify_with_qwen_max(source_list, candidate_list):
    """
    使用 qwen-max 校验原文列表和候选译文列表是否一一对应
    返回: True (通过) / False (不通过)
    """
    if len(source_list) != len(candidate_list):
        return False
        
    # 构造一个便于 AI 判断的 Prompt
    check_prompt = "请校验以下【原文列表】与【译文列表】是否内容对应，且行数一致。\n"
    check_prompt += "如果完全对应，请仅回复 YES。如果不对应或有错位，请回复 NO。\n\n"
    check_prompt += "【原文列表】:\n" + "\n".join(source_list) + "\n\n"
    check_prompt += "【译文列表】:\n" + "\n".join(candidate_list)
    
    try:
        completion = client.chat.completions.create(
            model="qwen-max", # 使用更强的模型做裁判
            messages=[{"role": "user", "content": check_prompt}]
        )
        answer = completion.choices[0].message.content.strip().upper()
        # 记录裁判的思考过程（可选，为了日志干净暂只记录结果）
        return "YES" in answer
    except Exception as e:
        logger.error(f"Verification API Error: {e}")
        return False
    

# --- 2. 初始化与断点续传加载 ---

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
    df_raw = pd.read_csv(INPUT_FILE, encoding='UTF-8')
    all_source_list = df_raw['物料描述'].astype(str).tolist()
    print(f"数据源加载成功，共 {len(all_source_list)} 条。")
except Exception as e:
    print(f"读取原始文件失败: {e}")
    exit()

history_data = [] 
inverted_index = collections.defaultdict(list) 

start_index = 0
if os.path.exists(OUTPUT_FILE):
    print("检测到历史进度文件，正在加载并构建索引...")
    try:
        df_done = pd.read_csv(OUTPUT_FILE)
        done_sources = df_done['物料描述'].astype(str).tolist()
        done_targets = df_done['英文描述'].astype(str).tolist()
        processed_count = len(df_done)
        
        for idx, (src, tgt) in enumerate(zip(done_sources, done_targets)):
            history_data.append((src, tgt))
            for w in jieba.cut(src):
                if len(w) >= 2:
                    inverted_index[w].append(idx)
        
        start_index = processed_count
        print(f"--- 成功恢复进度，已处理 {start_index} 条，将从第 {start_index + 1} 条开始 ---")
        
    except Exception as e:
        print(f"警告：读取进度文件失败 ({e})，将重新开始。")
else:
    print("--- 未发现进度文件，开始新任务 ---")
    pd.DataFrame(columns=['物料描述', '英文描述']).to_csv(OUTPUT_FILE, index=False, encoding='UTF-8')

# --- 3. 批处理循环 (带进度条与日志) ---

total_items = len(all_source_list)

# 使用 tqdm 创建进度条
# initial: 初始进度位置
# total: 总任务量
# unit: 单位名称
with tqdm(total=total_items, initial=start_index, unit="row", desc="Processing") as pbar:
    
    for i in range(start_index, total_items, BATCH_SIZE):
        batch_source = all_source_list[i : i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        # 使用 tqdm.write 代替 print，避免打乱进度条显示
        # tqdm.write(f"--- 处理批次 {current_batch_num} ---")
        
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

       # B. 调用 API (逻辑更新)
        for attempt in range(MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": input_text}]
                
                completion = client.chat.completions.create(
                    model="qwen-mt-plus",
                    messages=messages,
                    extra_body={"translation_options": translation_options}
                )
                
                response_text = completion.choices[0].message.content
                output_list = response_text.strip().split('\n')
                
                # --- LOG: 记录原始输出 ---
                logger.info(f"====== BATCH {current_batch_num} OUTPUT (Attempt {attempt+1}) ======\n{response_text}")

                # === 校验逻辑分支 ===
                
                # 情况 1: 行数完美匹配
                if len(output_list) == len(batch_source):
                    batch_results = output_list
                    success = True
                    break # 跳出重试循环
                
                # 情况 2: 输出行数 > 输入行数 (尝试截取修复)
                elif len(output_list) > len(batch_source):
                    logger.warning(f"Batch {current_batch_num} mismatch: Input {len(batch_source)} vs Output {len(output_list)}. Trying heuristic fix...")
                    
                    # 截取最后 N 行 (N = 输入行数)
                    candidate_fix = output_list[-len(batch_source):]
                    
                    # 调用 qwen-max 进行裁判
                    tqdm.write(f"  [校验] 检测到行数冗余，正在请求 qwen-max 校验截取结果...")
                    is_valid = verify_with_qwen_max(batch_source, candidate_fix)
                    
                    if is_valid:
                        logger.info(f"Batch {current_batch_num} heuristic fix VERIFIED by qwen-max. Accepted.")
                        tqdm.write(f"  √ qwen-max 校验通过，采纳修复结果")
                        batch_results = candidate_fix
                        success = True
                        break # 修复成功，跳出重试
                    else:
                        logger.warning(f"Batch {current_batch_num} heuristic fix REJECTED by qwen-max.")
                        tqdm.write(f"  X qwen-max 校验拒绝，继续重试")
                        # 校验失败，继续进入下一次 attempt 循环
                
                # 情况 3: 输出行数 < 输入行数 (无法修复，只能重试)
                else:
                    logger.warning(f"Batch {current_batch_num} mismatch: Input {len(batch_source)} vs Output {len(output_list)} (Too Short)")
                    tqdm.write(f"  [重试 {attempt+1}] 行数不足")
                    continue
            
            except Exception as e:
                logger.error(f"Batch {current_batch_num} Error (Attempt {attempt+1}): {e}")
                tqdm.write(f"  [重试 {attempt+1}] 错误: {e}")
                time.sleep(1)

        # C. 更新内存索引
        if success:
            current_start_idx_in_history = len(history_data)
            for idx, (src, tgt) in enumerate(zip(batch_source, batch_results)):
                history_data.append((src, tgt))
                abs_idx = current_start_idx_in_history + idx
                for w in jieba.cut(src):
                    if len(w) >= 2:
                        inverted_index[w].append(abs_idx)

        # D. 保存文件
        try:
            df_batch = pd.DataFrame({
                '物料描述': batch_source,
                '英文描述': batch_results
            })
            df_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='UTF-8')
        except Exception as e:
            logger.error(f"Save File Error: {e}")
            tqdm.write(f"  !!! 写入文件失败: {e}")
        
        # 更新进度条
        pbar.update(len(batch_source))

print("\n" + "="*30)
print("所有处理完毕！")
print(f"最终结果保存在: {OUTPUT_FILE}")
print(f"运行日志保存在: {LOG_FILE}")