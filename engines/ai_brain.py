"""
ai_brain.py — AI Market Brain
主力：Groq API（免費，llama-3.3-70b-versatile）
備用：Claude Sonnet
額外：自動生成 Prompt，可貼到任何 AI（ChatGPT / Claude / Gemini / Grok）
"""
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os


# ── Groq API ───────────────────────────────────────────────────────────────────

GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # 主力：最強免費模型
    "llama-3.1-8b-instant",      # 備用：快速輕量
    "mixtral-8x7b-32768",        # 備用：長上下文
]


def _get_secret(key: str) -> str:
    """Get API key from Streamlit secrets or env."""
    try:
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, "")


def _call_groq(prompt: str, system: str, max_tokens: int = 1000,
               model: str = None) -> tuple:
    """
    Call Groq API (free tier).
    Returns: (response_text, model_used)
    需要 GROQ_API_KEY 在 Streamlit Secrets
    """
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        return "", ""

    target_model = model or GROQ_MODELS[0]

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": target_model,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
            },
            timeout=30,
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            text = data["choices"][0]["message"]["content"]
            return text, target_model
        err = data.get("error", {}).get("message", str(data))
        # Rate limit — try next model
        if "rate" in err.lower() and len(GROQ_MODELS) > 1:
            return _call_groq(prompt, system, max_tokens, GROQ_MODELS[1])
        return "Groq Error: " + err, target_model
    except Exception as e:
        return "Groq Connection Error: " + str(e), target_model


def _call_claude(prompt: str, system: str, max_tokens: int = 1000) -> tuple:
    """Claude API fallback."""
    api_key = _get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return "", "claude-sonnet"

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        data = resp.json()
        if "content" in data and data["content"]:
            return data["content"][0].get("text", ""), "claude-sonnet-4"
        err = data.get("error", {}).get("message", "Unknown")
        return "Claude Error: " + err, "claude-sonnet-4"
    except Exception as e:
        return "Claude Error: " + str(e), "claude-sonnet-4"


def call_ai(prompt: str, system: str, max_tokens: int = 1000) -> tuple:
    """
    智能路由：優先 Groq（免費），失敗才用 Claude。
    Returns: (response_text, model_used)
    """
    # 1. 嘗試 Groq
    text, model = _call_groq(prompt, system, max_tokens)
    if text and not text.startswith("Groq"):
        return text, "🟢 Groq · " + model

    # 2. 嘗試 Claude
    text2, model2 = _call_claude(prompt, system, max_tokens)
    if text2 and not text2.startswith("Claude Error"):
        return text2, "🔵 " + model2

    # 3. 兩者都失敗 — 返回說明
    if text:
        return text, "❌ No valid API key"
    return "⚠️ 請在 Streamlit Secrets 設置 GROQ_API_KEY（免費）或 ANTHROPIC_API_KEY", "❌ No API"


# ── Market Context Builder ─────────────────────────────────────────────────────

def build_market_context(
    regime: dict,
    corr_matrix: pd.DataFrame,
    lead_lag: pd.DataFrame,
    contagion: pd.DataFrame,
    breakdown: pd.DataFrame,
    returns: pd.DataFrame,
) -> str:
    """從市場數據自動生成結構化 context 字串。"""
    lines = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append("=== Tesla Market Brain 數據快照 (" + now_str + ") ===")
    lines.append("")

    # Regime
    lines.append("【市場狀態】" + regime.get("current_name", "UNKNOWN") +
                 " (置信度: " + str(int(regime.get("confidence", 0) * 100)) + "%)")

    # TSLA performance
    if not returns.empty and "TSLA" in returns.columns:
        tsla_5d  = returns["TSLA"].tail(5).sum() * 100
        tsla_20d = returns["TSLA"].tail(20).sum() * 100
        tsla_vol = returns["TSLA"].tail(20).std() * np.sqrt(252) * 100
        lines.append(
            "【TSLA表現】5日: " + ("%+.2f" % tsla_5d) + "% | " +
            "20日: " + ("%+.2f" % tsla_20d) + "% | " +
            "年化波動: " + ("%.1f" % tsla_vol) + "%"
        )

    # Top correlations
    if not corr_matrix.empty and "TSLA" in corr_matrix.columns:
        tsla_corr = corr_matrix["TSLA"].drop("TSLA").sort_values(ascending=False)
        top_pos = ", ".join([s + "(" + ("%+.2f" % v) + ")" for s, v in tsla_corr.head(5).items()])
        top_neg = ", ".join([s + "(" + ("%+.2f" % v) + ")" for s, v in tsla_corr.tail(3).items()])
        lines.append("【正相關TOP5】" + top_pos)
        lines.append("【負相關TOP3】" + top_neg)

    # Lead-lag
    if not lead_lag.empty:
        leaders   = lead_lag[lead_lag["direction"] == "leads TSLA"].head(4)
        followers = lead_lag[lead_lag["direction"] == "follows TSLA"].head(3)
        if not leaders.empty:
            l_str = ", ".join([
                r["symbol"] + "(+" + str(r["best_lag_bars"]) + "bars,ρ=" + ("%.2f" % r["correlation"]) + ")"
                for _, r in leaders.iterrows()
            ])
            lines.append("【TSLA領先指標】" + l_str)
        if not followers.empty:
            f_str = ", ".join([r["symbol"] + "(" + str(r["best_lag_bars"]) + "bars)" for _, r in followers.iterrows()])
            lines.append("【TSLA跟隨者】" + f_str)

    # Contagion top 5
    if not contagion.empty:
        c_str = ", ".join([
            r["symbol"] + "(" + ("%+.1f" % r["expected_impact_pct"]) + "%,β=" + ("%.2f" % r["beta_to_tsla"]) + ")"
            for _, r in contagion.head(5).iterrows()
        ])
        lines.append("【TSLA-5%衝擊傳染】" + c_str)

    # Breakdown alerts
    if not breakdown.empty:
        breaking = breakdown[breakdown["signal"].str.contains("Breaking|Strengthen")]
        if not breaking.empty:
            b_list = []
            for _, r in breaking.head(4).iterrows():
                b_list.append(r["symbol"] + ":" + r["signal"] + "(Δρ=" + ("%+.3f" % r["delta"]) + ")")
            lines.append("【關聯異常】" + " | ".join(b_list))

    # Key asset 5D returns
    if not returns.empty:
        key_assets = ["NVDA", "QQQ", "SPY", "ARKK", "VIX", "BTC", "TLT"]
        available  = [a for a in key_assets if a in returns.columns]
        ret_strs   = [a + ":" + ("%+.1f" % (returns[a].tail(5).sum() * 100)) + "%" for a in available]
        lines.append("【關鍵資產5日】" + " | ".join(ret_strs))

    return "\n".join(lines)


# ── Prompt Templates ───────────────────────────────────────────────────────────

PROMPT_TYPES = {
    "market_brain":     "🧠 每日市場診斷",
    "tsla_trade":       "🎯 TSLA 交易策略",
    "regime_analysis":  "🏛️ 市場狀態深度分析",
    "narrative_chain":  "🌊 Elon 敘事傳播鏈",
    "risk_report":      "⚠️ 風險評估報告",
    "sector_rotation":  "🔄 板塊輪動分析",
    "tweet_impact":     "⚡ 推文市場衝擊",
}

TARGET_AIS = ["通用", "ChatGPT", "Claude", "Gemini", "Grok", "Perplexity"]

SYSTEM_ROLES = {
    "market_brain":    "你是對沖基金的首席量化分析師，專精美股市場結構分析。",
    "tsla_trade":      "你是 Tesla 專業交易員，擅長日內交易和波段操作策略。",
    "regime_analysis": "你是市場結構研究員，專精市場狀態識別和板塊輪動。",
    "narrative_chain": "你是量化金融研究員，專門研究 Elon Musk 敘事對市場的傳播效應。",
    "risk_report":     "你是風險管理師，專精系統性風險評估和資產組合保護。",
    "sector_rotation": "你是板塊輪動策略師，擅長識別資金流動方向。",
    "tweet_impact":    "你是社交媒體量化分析師，專門分析名人言論對金融市場的影響。",
}


def generate_prompt(prompt_type: str, context: str,
                    language: str = "繁體中文", extra: str = "") -> str:
    """生成完整 AI Prompt（可複製貼到任何 AI）。"""

    role = SYSTEM_ROLES.get(prompt_type, SYSTEM_ROLES["market_brain"])

    extra_block = ""
    if extra:
        extra_block = "\n補充信息：" + extra + "\n"

    templates = {

        "market_brain": (
            "基於以下實時市場數據，生成今日市場智能簡報：\n\n" + context + extra_block + "\n"
            "請輸出（用" + language + "）：\n\n"
            "🧠 **市場核心診斷**\n（一句話說明今日市場最重要的結構特徵）\n\n"
            "📊 **TSLA 定位分析**\n- 當前角色：Leader 還是 Follower？\n- 原因：（基於相關性和 Lead-Lag 數據）\n\n"
            "💰 **資金流向判斷**\n（哪個板塊/資產正在吸引資金？離開哪裡？）\n\n"
            "⚡ **最重要異常信號**\n（哪個數據最不尋常？為什麼值得注意？）\n\n"
            "🎯 **板塊輪動機會**\n（哪個板塊正在或即將接力？）\n\n"
            "⚠️ **主要下行風險**\n（最需要警惕的 1-2 個風險因子）\n\n"
            "🎬 **操作思路參考**（非投資建議）\n（具體的進出場思路框架）\n\n"
            "要求：每點 1-3 句，精準有力，有數據支撐。"
        ),

        "tsla_trade": (
            "基於以下市場數據，提供 TSLA 交易策略分析：\n\n" + context + extra_block + "\n"
            "請輸出（用" + language + "）：\n\n"
            "📍 **TSLA 當前技術環境**\n（根據相關性和市場狀態判斷多空偏向）\n\n"
            "🎯 **核心交易邏輯**\n（為什麼現在是做多/做空/觀望？）\n\n"
            "⚡ **觸發信號**\n- 做多觸發：\n- 做空觸發：\n\n"
            "🛡️ **風險控制**\n- 止損參考：\n- 倉位建議：\n- 時間框架：\n\n"
            "💡 **領先指標監控**\n（根據Lead-Lag分析，應重點監控哪些資產的動向？）\n\n"
            "⚠️ 本分析僅供研究參考，不構成投資建議。"
        ),

        "regime_analysis": (
            "深度分析以下市場狀態數據：\n\n" + context + extra_block + "\n"
            "請分析（用" + language + "）：\n\n"
            "🏛️ **當前市場 Regime**\n（詳述當前狀態的特徵和形成原因）\n\n"
            "📊 **關鍵結構特徵**\n（相關性矩陣揭示了什麼市場結構？）\n\n"
            "🔄 **Regime 轉換信號**\n- 轉向 Risk-On 的信號：\n- 轉向 Risk-Off 的信號：\n\n"
            "🌊 **資金流動路徑**\n（當前 Regime 下資金通常如何流動？）\n\n"
            "💎 **歷史類比**\n（這個市場狀態類似哪些歷史時期？如何演化？）\n\n"
            "🎯 **策略建議**\n（在此 Regime 下，哪類交易策略有優勢？）"
        ),

        "narrative_chain": (
            "分析 Elon Musk 敘事對市場的傳播鏈：\n\n市場數據背景：\n" + context + extra_block + "\n"
            "請分析（用" + language + "）：\n\n"
            "📌 **敘事分類**\n（AI/FSD/Optimus/Crypto/Macro/Political？）\n\n"
            "🌊 **市場傳播鏈**\n"
            "Elon 敘事 → 直接影響 → 二階影響 → 三階影響\n"
            "例如：Tweet → TSLA(直接) → ARKK/NVDA(科技情緒) → QQQ/BTC(風險偏好)\n\n"
            "⏱️ **時間衰減模型**\n- T+0（即時）：哪些資產立即反應？\n"
            "- T+1小時：誰跟隨？\n- T+1天：影響持續或反轉？\n\n"
            "📊 **當前市場相關性強化效應**\n（根據數據，Elon 敘事在現在的市場結構下影響力有多大？）\n\n"
            "🎯 **交易機會識別**\n（如何利用敘事傳播鏈的時間差套利？）"
        ),

        "risk_report": (
            "生成 TSLA 相關風險評估報告：\n\n" + context + extra_block + "\n"
            "請輸出（用" + language + "）：\n\n"
            "🚨 **系統性風險評估**（1-10分）\n- 市場整體風險：/10\n- TSLA 特定風險：/10\n- 傳染風險：/10\n\n"
            "💣 **主要風險因子**\n1. 最高優先級：\n2. 次要風險：\n3. 尾部風險：\n\n"
            "🌊 **風險傳染路徑**\n（根據 Beta 數據，TSLA 下跌如何傳染？誰先中招？）\n\n"
            "🛡️ **對沖建議**\n- 直接對沖：（如 TSLA Put / UVXY）\n"
            "- 相關性對沖：（利用負相關資產）\n- 宏觀對沖：（TLT/GLD等）\n\n"
            "📊 **壓力測試場景**\n- 基本場景（-5%）：影響評估\n"
            "- 熊市場景（-15%）：影響評估\n- 崩盤場景（-30%）：影響評估"
        ),

        "sector_rotation": (
            "分析當前板塊輪動動態：\n\n" + context + extra_block + "\n"
            "請輸出（用" + language + "）：\n\n"
            "🔄 **當前輪動狀態**\n（哪個板塊正在領漲？哪個在退潮？）\n\n"
            "💰 **資金流向地圖**\n流出：→ 流入：\n（根據相關性變化和 ETF 表現推斷）\n\n"
            "⚡ **TSLA 在輪動中的角色**\n（TSLA 是這次輪動的受益者還是受害者？）\n\n"
            "🎯 **下一個接力板塊**\n（根據歷史輪動規律和當前數據，誰最可能接力？）\n\n"
            "📊 **ETF 監控清單**\n（哪些 ETF 的異動最值得關注？）\n\n"
            "🕐 **時間窗口估計**\n（當前輪動預計持續多久？轉折信號是什麼？）"
        ),

        "tweet_impact": (
            "分析以下推文/新聞對市場的影響：\n\n市場背景：\n" + context + "\n\n"
            "分析對象：\n「" + extra + "」\n\n"
            "請輸出（用" + language + "）：\n\n"
            "📌 **內容分類**\n主題：（AI/FSD/Crypto/Macro/Political/Tesla產品）\n"
            "情緒：（-10到+10）\n緊急程度：（低/中/高/極高）\n\n"
            "📈 **直接市場衝擊**\n- TSLA預期反應：方向 + 幅度估計\n- 觸發時間：（即時/開盤/延遲反應）\n\n"
            "🌊 **傳播鏈分析**\n第一波（0-5分鐘）：\n第二波（5-60分鐘）：\n第三波（1天+）：\n\n"
            "📊 **根據當前相關性結構**\n最可能受影響的3個資產：\n可能被錯誤定價的機會：\n\n"
            "⚡ **交易窗口**\n（如何在敘事傳播過程中找到機會？）\n\n"
            "⚠️ **反向風險**\n（如果市場反應超預期或相反，應如何應對？）"
        ),
    }

    task = templates.get(prompt_type, templates["market_brain"])

    return "[系統角色]\n" + role + "\n\n[任務]\n" + task


def get_prompt_for_ai(prompt_type: str, context: str,
                       target_ai: str = "通用", language: str = "繁體中文",
                       extra: str = "") -> str:
    """生成針對特定 AI 平台優化的 Prompt。"""
    base = generate_prompt(prompt_type, context, language, extra)

    ai_preambles = {
        "ChatGPT":    "請扮演以下角色並完成任務。如需使用工具分析，請先說明你的分析框架。\n\n",
        "Claude":     "請仔細閱讀數據後再回答，避免幻覺，如有不確定請說明。\n\n",
        "Gemini":     "Please analyze the following financial data carefully. Respond in Traditional Chinese.\n\n",
        "Grok":       "分析以下市場數據，提供你獨特的量化視角。不要客套，直接說重點。\n\n",
        "Perplexity": "Based on the following financial data, provide a research-grade analysis:\n\n",
        "通用":       "",
    }

    preamble = ai_preambles.get(target_ai, "")
    return preamble + base


# ── 主要 AI 函數（供 app.py 調用）────────────────────────────────────────────

def generate_market_brain(regime, corr_matrix, lead_lag, contagion, breakdown, returns) -> tuple:
    """
    生成市場分析。
    Returns: (analysis_text, model_used)
    """
    context = build_market_context(regime, corr_matrix, lead_lag, contagion, breakdown, returns)
    prompt  = generate_prompt("market_brain", context)
    system  = (
        "你是 Tesla Market Brain，一位對沖基金級別的 AI 量化分析師。"
        "用繁體中文回答。簡潔、精準、有洞察力。每點1-2句，突出異常信號，有明確交易含義。"
    )
    return call_ai(prompt, system, max_tokens=900)


def analyze_narrative_impact(tweet_text: str, context: str = "") -> tuple:
    """分析 Elon 推文 / 新聞標題的市場衝擊。"""
    prompt = generate_prompt("tweet_impact", context or "（無額外市場背景）", extra=tweet_text)
    system = "你是量化分析師，專門分析社交媒體對金融市場的影響。用繁體中文，簡潔精準。"
    return call_ai(prompt, system, max_tokens=600)


def score_sentiment_simple(text: str) -> dict:
    """規則式情緒評分（不需要 API）。"""
    text_lower = text.lower()
    bull_words  = ["bullish", "moon", "surge", "rally", "buy", "long", "breakthrough",
                   "innovation", "launch", "record", "profit", "growth", "strong",
                   "上漲", "看漲", "突破", "創新", "增長"]
    bear_words  = ["crash", "dump", "sell", "short", "down", "fail", "loss", "risk",
                   "concern", "warning", "下跌", "看跌", "崩盤", "風險", "警告"]
    urgent_words= ["breaking", "just", "now", "alert", "urgent", "immediately",
                   "突破", "剛剛", "緊急", "馬上"]
    categories  = {
        "AI/xAI":       ["ai", "grok", "xai", "machine learning", "artificial"],
        "FSD/Robotaxi": ["fsd", "robotaxi", "autopilot", "self-driving", "autonomy"],
        "Optimus":      ["optimus", "robot", "humanoid"],
        "Crypto":       ["bitcoin", "btc", "crypto", "doge", "dogecoin", "ethereum"],
        "Macro":        ["rate", "fed", "inflation", "economy", "gdp", "tariff", "trade"],
        "Political":    ["government", "president", "congress", "policy", "doge dept", "epa"],
        "Tesla":        ["tesla", "tsla", "model", "cybertruck", "gigafactory"],
    }
    bull_count   = sum(1 for w in bull_words   if w in text_lower)
    bear_count   = sum(1 for w in bear_words   if w in text_lower)
    urgent_count = sum(1 for w in urgent_words if w in text_lower)
    total  = bull_count + bear_count
    score  = (bull_count - bear_count) / max(total, 1)
    urgency= "High" if urgent_count >= 2 else ("Medium" if urgent_count == 1 else "Low")
    detected = [cat for cat, words in categories.items() if any(w in text_lower for w in words)]
    category = detected[0] if detected else "General"
    return {
        "score":          round(score, 2),
        "urgency":        urgency,
        "category":       category,
        "bull_signals":   bull_count,
        "bear_signals":   bear_count,
        "sentiment_label":"🟢 Bullish" if score > 0.1 else ("🔴 Bearish" if score < -0.1 else "⚪ Neutral"),
    }
