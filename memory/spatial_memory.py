"""
空间记忆检索
- 从 data/space/spatial_memory.json 加载物品列表
- 使用 BGE 向量计算语义相似度（似然）
- 结合 weight×confidence 先验计算后验分数（乘法框架）
- 返回最优结果及语气等级
"""
import json
import os
from typing import Optional
import numpy as np
import config
from memory.semantic_memory import get_model   # 共享 BGE 模型单例

# ==================== 算法参数 ====================
FEATURES_FLOOR = 0.2      # features 相似度下限
REFS_FLOOR = 0.15         # references 相似度下限
NAME_BOOST_THRESHOLD = 0.7   # name 超过此阈值激活加分
NAME_BOOST_BONUS = 0.15      # name 加分值

# ==================== 数据加载 ====================
def load_spatial_memories() -> list[dict]:
    """加载全量空间记忆，并自动为缺失向量的物品补全向量"""
    if not os.path.exists(config.SPATIAL_MEMORY_FILE):
        return []
    with open(config.SPATIAL_MEMORY_FILE, 'r', encoding='utf-8') as f:
        items = json.load(f)

    # 检查是否有物品缺少向量，并补全
    model = None  # 延迟加载
    dirty = False
    for item in items:
        name_vec = item.get('name_vec')
        features_vec = item.get('features_vec')
        refs_vec = item.get('refs_vec')
        # 如果三个向量有一个缺失或为空，就需要计算
        if not name_vec or not features_vec or not refs_vec:
            if model is None:
                model = get_model()
            name = item.get('name', '')
            features = item.get('features', '')
            refs = item.get('references', [])
            refs_text = '，'.join(refs) if refs else ''

            item['name_vec'] = model.encode(name, normalize_embeddings=True).tolist() if name else []
            item['features_vec'] = model.encode(features, normalize_embeddings=True).tolist() if features else []
            item['refs_vec'] = model.encode(refs_text, normalize_embeddings=True).tolist() if refs_text else []
            dirty = True

    # 如果有补全，写回文件
    if dirty:
        with open(config.SPATIAL_MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"[空间记忆] 已为 {sum(1 for _ in items if _['name_vec'])} 个物品补全向量并保存")

    return items


# ==================== 检索算法 ====================
def search_spatial_memory(user_query: str) -> Optional[dict]:
    """
    空间记忆检索：乘法似然 × 先验
    """
    items = load_spatial_memories()
    if not items:
        return None

    model = get_model()
    query_vec = model.encode(user_query, normalize_embeddings=True)

    print(f"\n[空间记忆检索] 查询: \"{user_query}\"")
    print(f"共加载 {len(items)} 个物品，开始计算得分...\n")

    scored = []
    for idx, item in enumerate(items, start = 1):           # 索引从1开始
        name = item.get('name', '')             # 物品名称
        features = item.get('features', '')     # 物品特征
        refs = item.get('references', [])       # 物品参考
        references_text = '，'.join(refs) if refs else ''   # 拼接参照物
        confidence = item.get('confidence', 0.5)# 物品置信度
        weight = item.get('weight', 1.0)        # 物品权重

        # --- 优先使用预存向量，没有则实时编码 ---
        if 'name_vec' in item and item['name_vec']:
            name_vec = np.array(item['name_vec'])
            sim_name = float(np.dot(query_vec, name_vec))
        else:
            sim_name = _cosine_sim(query_vec, model, name)

        if 'features_vec' in item and item['features_vec']:
            features_vec = np.array(item['features_vec'])
            sim_features = float(np.dot(query_vec, features_vec))
        else:
            sim_features = _cosine_sim(query_vec, model, features)

        if 'refs_vec' in item and item['refs_vec']:
            refs_vec = np.array(item['refs_vec'])
            sim_refs = float(np.dot(query_vec, refs_vec))
        else:
            sim_refs = _cosine_sim(query_vec, model, references_text)





        # --- 似然（乘法 + 硬下限 + 可选加分） ---
        # features 下限
        sim_features_final = max(sim_features, FEATURES_FLOOR)
        features_floor_active = sim_features < FEATURES_FLOOR       # 是否激活下限保护

        # references 下限
        sim_refs_final = max(sim_refs, REFS_FLOOR)
        refs_floor_active = sim_refs < REFS_FLOOR

        #todo name 加分（超过阈值时）
        sim_name_boosted = sim_name
        bonus_active = False            # 是否激活加分
        if sim_name >= NAME_BOOST_THRESHOLD:
            sim_name_boosted = min(sim_name + NAME_BOOST_BONUS, 0.99)  # 封顶0.99
            bonus_active = True

        # 乘法似然
        likelihood = sim_name_boosted * sim_features_final * sim_refs_final

        # 先验
        prior = round(weight * confidence, 4)

        # 后验
        posterior = prior * likelihood

        # 可视化打印
        print(f"--- 物品{idx}: {name} (空间{item.get('space_id', 0)}) ---")
        print(f"  name 相似度: {sim_name:.3f}", end="")
        if bonus_active:
            print(f" -> 加分后 {sim_name_boosted:.3f}", end="")
        print()
        print(f"  features 相似度: {sim_features:.3f}", end="")
        if features_floor_active:
            print(f" -> 下限保护 {sim_features_final:.3f}", end="")
        print()
        print(f"  references 相似度: {sim_refs:.3f}", end="")
        if refs_floor_active:
            print(f" -> 下限保护 {sim_refs_final:.3f}", end="")
        print()
        print(f"  似然 = {sim_name_boosted:.3f} × {sim_features_final:.3f} × {sim_refs_final:.3f} = {likelihood:.4f}")
        print(f"  先验 weight({weight}) × confidence({confidence}) = {prior}")
        print(f"  后验 = {prior} × {likelihood:.4f} = {posterior:.4f}")

        scored.append({
            'item': item,
            'score': posterior,
            'likelihood': round(likelihood, 4),
            'prior': prior,
            'sim_name': sim_name,
            'sim_features': sim_features,
            'sim_refs': sim_refs
        })

    if not scored:
        return None

    # 归一化（min-max）用于语气判定
    scores = [r['score'] for r in scored]
    min_s, max_s = min(scores), max(scores)
    if max_s > min_s:
        for r in scored:
            r['norm'] = (r['score'] - min_s) / (max_s - min_s)
    else:
        for r in scored:
            r['norm'] = 0.0

    # 取最高分
    scored.sort(key=lambda r: r['score'], reverse=True)
    best = scored[0]

    # 语气判定
    ns = best['norm']
    if ns >= 0.7:
        tone = 'high'
    elif ns >= 0.4:
        tone = 'mid'
    else:
        tone = 'low'

    print(f"\n===== 最优结果 =====")
    print(f"物品: {best['item']['name']}")
    print(f"后验得分: {best['score']:.4f} (归一化: {ns:.3f})")
    print(f"语气: {tone}")
    print(f"似然: {best['likelihood']:.4f}, 先验: {best['prior']}")
    print()

    return {
        'item': best['item'],
        'score': round(best['score'], 4),
        'tone': tone,
        'likelihood': best['likelihood'],
        'prior': best['prior']
    }


def _cosine_sim(query_vec: np.ndarray, model, text: str) -> float:
    """编码文本并返回与查询向量的余弦相似度"""
    if not text or not text.strip():
        return 0.0
    text_vec = model.encode(text, normalize_embeddings=True)
    return float(np.dot(query_vec, text_vec))


def format_spatial_result(result: dict) -> str:
    if not result:
        return "未找到相关物品。"
    item = result['item']
    name = item.get('name', '未知')
    located = item.get('located', [0, 0])
    refs = item.get('references', [])
    tone = result['tone']
    # 语气映射
    tone_text = {"high": "高", "mid": "中", "low": "低"}.get(tone, "中")
    ref_str = '、'.join(refs) if refs else '无'
    return (
        f"物品「{name}」，位于空间{item.get('space_id', '?')}，"
        f"坐标({located[0]:.1f}, {located[1]:.1f})，"
        f"附近参照物：{ref_str}。置信度：{tone_text}。"
    )

# ==================== Tool 入口（供 web_ui 调用） ====================
def run_spatial_tool(user_query: str) -> Optional[str]:
    """
    空间记忆 Tool 入口
    返回格式化文本 或 None（无匹配）
    """
    result = search_spatial_memory(user_query)
    if result is None:
        return "【环境记忆】当前没有任何物品记录，或未找到与您问题相关的物品。"
    return format_spatial_result(result)