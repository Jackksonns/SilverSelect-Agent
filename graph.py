import os
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pymysql

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# ============ 基础加载 ============
load_dotenv(override=True)

# 搜索工具（保留）
search_tool = TavilySearch(max_results=5, topic="general")

# ============ SQL 查询工具（新增安全闸口：仅允许 SELECT） ============
class SQLQuerySchema(BaseModel):
    sql_query: str = Field(description="仅允许 SELECT 的 MySQL 查询语句")

@tool(args_schema=SQLQuerySchema)
def sql_inter(sql_query: str) -> str:
    """只做 SELECT 查询，返回 JSON 字符串。"""
    q = sql_query.strip().lower()
    if not q.startswith("select"):
        return "❌ 仅允许执行 SELECT 语句，请修改后重试。"

    load_dotenv(override=True)
    connection = pymysql.connect(
        host=os.getenv('HOST'),
        user=os.getenv('USER'),
        passwd=os.getenv('MYSQL_PW'),
        db=os.getenv('DB_NAME'),
        port=int(os.getenv('PORT')),
        charset='utf8'
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            results = cursor.fetchall()
    finally:
        connection.close()
    return json.dumps(results, ensure_ascii=False)

# ============ Extract：提取数据为 pandas（同样仅允许 SELECT） ============
class ExtractQuerySchema(BaseModel):
    sql_query: str = Field(description="用于从 MySQL 提取数据的 SELECT 语句")
    df_name: str = Field(description="保存为 pandas 变量名，如 'df_streams'")

@tool(args_schema=ExtractQuerySchema)
def extract_data(sql_query: str, df_name: str) -> str:
    """将 SELECT 结果提取为 pandas.DataFrame 并保存为全局变量。"""
    if not sql_query.strip().lower().startswith("select"):
        return "❌ 仅允许执行 SELECT 语句，请修改后重试。"

    load_dotenv(override=True)
    connection = pymysql.connect(
        host=os.getenv('HOST'),
        user=os.getenv('USER'),
        passwd=os.getenv('MYSQL_PW'),
        db=os.getenv('DB_NAME'),
        port=int(os.getenv('PORT')),
        charset='utf8'
    )
    try:
        df = pd.read_sql(sql_query, connection)
        globals()[df_name] = df
        return f"✅ 已创建 DataFrame `{df_name}`，shape={df.shape}"
    except Exception as e:
        return f"❌ 执行失败：{e}"
    finally:
        connection.close()

# ============ Python 执行（非绘图） ============
class PythonCodeInput(BaseModel):
    py_code: str = Field(description="一段合法的 Python 代码字符串")

@tool(args_schema=PythonCodeInput)
def python_inter(py_code):
    """执行非绘图 Python 代码并返回输出或新变量。"""
    g = globals()
    try:
        return str(eval(py_code, g))
    except Exception:
        global_vars_before = set(g.keys())
        try:
            exec(py_code, g)
        except Exception as e:
            return f"代码执行时报错 {e}"
        global_vars_after = set(g.keys())
        new_vars = global_vars_after - global_vars_before
        if new_vars:
            result = {var: g[var] for var in new_vars}
            return str(result)
        return "已经顺利执行代码"

# ============ 绘图（可移植路径：IMAGES_DIR 或 ./images） ============
class FigCodeInput(BaseModel):
    py_code: str = Field(description="matplotlib/seaborn 绘图代码，需创建 fig 对象")
    fname: str = Field(description="图像对象变量名，如 'fig'，将保存为 images/fname.png")

@tool(args_schema=FigCodeInput)
def fig_inter(py_code: str, fname: str) -> str:
    """执行绘图代码并将图片保存到 IMAGES_DIR（默认 ./images）。"""
    current_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    local_vars = {"plt": plt, "pd": pd, "sns": sns, "np": np}

    # 允许通过环境变量控制输出目录
    images_dir = os.getenv("IMAGES_DIR", "./images")
    os.makedirs(images_dir, exist_ok=True)

    try:
        g = globals()
        exec(py_code, g, local_vars)
        g.update(local_vars)
        fig = local_vars.get(fname, None)
        if fig:
            image_filename = f"{fname}.png"
            abs_path = os.path.join(images_dir, image_filename)
            fig.savefig(abs_path, bbox_inches='tight')
            return f"✅ 图片已保存：{abs_path}"
        else:
            return "⚠️ 未找到 fig 对象，请确认变量名正确。"
    except Exception as e:
        return f"❌ 执行失败：{e}"
    finally:
        plt.close('all')
        matplotlib.use(current_backend)

# ============ 新增：竞争力评分（SilverRank：熵权 + TOPSIS） ============
class SilverRankInput(BaseModel):
    df_name: str = Field(description="包含指标的数据表 DataFrame 名称")
    pos_cols: list[str] = Field(description="正向指标列名列表，如 ['view_per_live','conversion_rate']")
    neg_cols: list[str] = Field(description="逆向指标列名列表，如 ['return_rate']")
    topk: int = Field(default=20, description="返回前K名")

def _min_max_norm(x):
    x = np.array(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn == 0:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def _entropy_weights(X):
    # X: ndarray shape (n_samples, n_features), 列已归一化在[0,1]
    X = np.clip(X, 1e-12, 1)  # 避免 log(0)
    P = X / (X.sum(axis=0, keepdims=True) + 1e-12)
    E = - (P * np.log(P)).sum(axis=0) / np.log(X.shape[0])
    d = 1 - E
    w = d / d.sum() if d.sum() > 0 else np.ones_like(d) / len(d)
    return w

@tool(args_schema=SilverRankInput)
def silverrank_score(df_name: str, pos_cols: list[str], neg_cols: list[str], topk: int = 20) -> str:
    """
    计算竞争力评分（SilverRank）：
    1) 正向/逆向指标分开归一化（逆向取1-归一化值）
    2) 熵权法计算权重
    3) TOPSIS 计算贴近理想解得分
    返回：权重表、总分TopK、每列Min/Max便于口径说明
    """
    if df_name not in globals():
        return f"❌ 未找到 DataFrame `{df_name}`，请先使用 extract_data 加载。"

    df = globals()[df_name].copy()
    cols = pos_cols + neg_cols
    for c in cols:
        if c not in df.columns:
            return f"❌ 列 {c} 不在 `{df_name}` 中。"

    # 归一化
    pos_mat = np.column_stack([_min_max_norm(df[c].values) for c in pos_cols]) if pos_cols else np.empty((len(df),0))
    neg_mat = np.column_stack([1 - _min_max_norm(df[c].values) for c in neg_cols]) if neg_cols else np.empty((len(df),0))
    X = np.column_stack([pos_mat, neg_mat]) if pos_cols or neg_cols else None
    if X is None:
        return "❌ 未指定任何指标列。"

    # 熵权
    w = _entropy_weights(X)

    # TOPSIS
    ideal_best = X.max(axis=0)
    ideal_worst = X.min(axis=0)
    d_pos = np.linalg.norm((X - ideal_best) * w, axis=1)
    d_neg = np.linalg.norm((X - ideal_worst) * w, axis=1)
    score = d_neg / (d_pos + d_neg + 1e-12)

    df_out = df.copy()
    df_out["silverrank_score"] = score
    top = df_out.sort_values("silverrank_score", ascending=False).head(topk)

    weights = pd.DataFrame({
        "metric": pos_cols + neg_cols,
        "direction": ["positive"]*len(pos_cols) + ["negative"]*len(neg_cols),
        "weight": w
    })

    # 指标口径回显（min/max）
    mm = pd.DataFrame({
        "metric": cols,
        "min": [df[c].min() for c in cols],
        "max": [df[c].max() for c in cols]
    })

    return json.dumps({
        "weights": weights.to_dict(orient="records"),
        "topk": top.to_dict(orient="records"),
        "minmax": mm.to_dict(orient="records")
    }, ensure_ascii=False)

# ============ 新增：建议卡（周度三条，可读短句 + 信号灯） ============
class SuggestCardsInput(BaseModel):
    df_rank_name: str = Field(description="包含 silverrank_score 的 DataFrame 名称")
    k: int = Field(default=3, description="建议条数，默认3")

@tool(args_schema=SuggestCardsInput)
def suggest_cards(df_rank_name: str, k: int = 3) -> str:
    """
    生成本周三条建议卡（短句 + 信号灯）：
    - 主推品类占比（依据Top样本的品类/地域命中度）
    - 建议价格带（依据客单价分位）
    - 上新清单与脚本要点（依据GPM/转化与退货率阈值）
    返回 JSON：list[{text, signal}]
    """
    if df_rank_name not in globals():
        return f"❌ 未找到 DataFrame `{df_rank_name}`。"
    df = globals()[df_rank_name].copy()
    if "silverrank_score" not in df.columns:
        return "❌ 需要先计算 silverrank_score。"

    out = []
    # 伪示例规则（可按你们库字段替换：region_hit, cate_hit, aov, return_rate, conversion）
    # 1) 主推品类占比
    text1 = "本周主推：提高命中度最高的品类占比到 35%-45%。"
    signal1 = "🟢"  # 可根据 cate_hit 均值设定阈值
    out.append({"text": text1, "signal": signal1})

    # 2) 价格带
    text2 = "建议价格带：聚焦近两周客单价中位数 ±10% 区间。"
    signal2 = "🟡"
    out.append({"text": text2, "signal": signal2})

    # 3) 上新与脚本
    text3 = "上新与脚本：首屏亮点 + 两段权益触达，避免连续三场同款疲劳。"
    signal3 = "🟢"
    out.append({"text": text3, "signal": signal3})

    return json.dumps(out, ensure_ascii=False)

# ============ 新增：供应商撮合（白名单 + 多维排序） ============
class MatchSuppliersInput(BaseModel):
    suppliers_df: str = Field(description="供应商表 DataFrame 名")
    region: str = Field(description="主播主要地域，如 '华南'")
    category: str = Field(description="主推品类，如 '果蔬生鲜'")
    price_min: float = Field(description="建议价格下限", default=0.0)
    price_max: float = Field(description="建议价格上限", default=99999.0)
    topn: int = Field(description="返回前N条", default=10)

@tool(args_schema=MatchSuppliersInput)
def match_suppliers(suppliers_df: str, region: str, category: str,
                    price_min: float = 0.0, price_max: float = 99999.0, topn: int = 10) -> str:
    """
    对白名单供应商进行多维筛选与排序：
    - 基于 region/category/价格带 过滤
    - 信誉评分优先、发货时效与历史退货率加权
    返回 JSON：可对接清单（含预警项）
    """
    if suppliers_df not in globals():
        return f"❌ 未找到 DataFrame `{suppliers_df}`。"
    df = globals()[suppliers_df].copy()

    # 预期字段（可按你们实际列名对接）：region, category, price_min, price_max, rating, is_whitelist, sla_hrs, return_rate
    for c in ["region","category","price_min","price_max","rating","is_whitelist","sla_hrs","return_rate"]:
        if c not in df.columns:
            return f"❌ 供应商表缺少列：{c}"

    cond = (
        (df["is_whitelist"]==1) &
        (df["category"]==category) &
        (df["region"].str.contains(region)) &
        (df["price_min"]<=price_max) &
        (df["price_max"]>=price_min)
    )
    cand = df.loc[cond].copy()
    if cand.empty:
        return "⚠️ 未匹配到供应商，请放宽条件重试。"

    # 简单打分：rating 为主，sla_hrs 越小越好，return_rate 越低越好
    cand["score"] = cand["rating"] - 0.001*cand["sla_hrs"] - 0.5*cand["return_rate"]
    cand = cand.sort_values("score", ascending=False).head(topn)

    # 生成预警文字（适老化短句）
    def warn(row):
        w = []
        if row["sla_hrs"]>72: w.append("发货>72h")
        if row["return_rate"]>0.12: w.append("退货>12%")
        return "、".join(w) if w else "无"
    cand["warning"] = cand.apply(warn, axis=1)

    return cand.to_json(orient="records", force_ascii=False)

# ============ 项目专用提示词（适老化风格 + 口径一致） ============
prompt = """
你是“银发甄选”的智能体（数据参谋 + 供应经理）。
服务对象：银发主播与农特产供应端。场景：农品直播的选品、撮合、复盘。

【数据口径】
仅使用十项以内核心指标：场均观看、活跃粉丝、观看GPM、曝光GPM、转化率、客单价、退货率、地域命中、品类命中、7日内首单。
所有分析均以 SELECT 查询与 DataFrame 为基础；数字尽量保留 1 位小数并标注样本期。

【输出规范（适老化）】
- 结论先行，每条 ≤ 24 字，一次最多 3 条。
- 用“建议卡”返回短句，并给出 🟢🟡🔴 信号灯（绿=执行、黄=小心、红=暂缓）。
- 尽量避免专业黑话，给“做法”不是“概念”。

【工作流】
1) 如需取数：先调用 extract_data / sql_inter（仅 SELECT）。
2) 计算竞争力：调用 silverrank_score（熵权 + TOPSIS），形成 SilverRank 评分卡。
3) 生成建议：调用 suggest_cards，输出“三条本周建议”。
4) 对接供给：调用 match_suppliers 给出 Top-N 可对接清单与预警。
5) 如需出图：调用 fig_inter 生成“指标总览/前后对比”图。
6) 回答最后给出“下一步行动”三条短句。

【禁止事项】
- 不执行非 SELECT 的 SQL；不暴露隐私字段；不杜撰数据与路径。

请严格按上述口径完成用户请求。
"""

# 工具列表（加入新工具）
tools = [search_tool, python_inter, fig_inter, sql_inter, extract_data,
         silverrank_score, suggest_cards, match_suppliers]

# 模型与 Agent
model = ChatDeepSeek(model="deepseek-chat")
graph = create_react_agent(model=model, tools=tools, prompt=prompt)
