# strategy_scoring_patch.py
# notify.py에 붙여 넣어 사용할 수 있는 5% / 30% 전략 점수 계산 모듈
# 전제: df 컬럼은 o, h, l, c, v 를 사용합니다. 날짜는 df.index에 있어야 합니다.

import pandas as pd
import numpy as np


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(df, period=14):
    high = df["h"]
    low = df["l"]
    close = df["c"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()


def alpha_heikin_ashi(df, smooth=8):
    def ema(s, p):
        return s.ewm(span=p, adjust=False).mean()

    o = ema(df["o"], smooth)
    h = ema(df["h"], smooth)
    l = ema(df["l"], smooth)
    c = ema(df["c"], smooth)

    ha_close = (o + h + l + c) / 4
    ha_open = ha_close.copy()

    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)

    out = pd.DataFrame(index=df.index)
    out["ha_o"] = ha_open
    out["ha_h"] = ha_high
    out["ha_l"] = ha_low
    out["ha_c"] = ha_close
    out["ha_bull"] = out["ha_c"] > out["ha_o"]
    out["ha_bear"] = out["ha_c"] < out["ha_o"]
    out["ha_body_pct"] = ((out["ha_c"] - out["ha_o"]).abs() / (out["ha_h"] - out["ha_l"]).replace(0, np.nan)) * 100
    return out


def is_ha_bull_turn(ha):
    if len(ha) < 2:
        return False
    return bool((ha["ha_bear"].iloc[-2]) and (ha["ha_bull"].iloc[-1]))


def has_recent_doji(ha, lookback=3, body_max_pct=18):
    recent = ha.tail(lookback)
    return bool((recent["ha_body_pct"] <= body_max_pct).any())


def weekly_uptrend(df):
    # 주봉 종가가 5주선 위 + 5주선이 10주선 위면 상승 추세로 판단
    w = df.resample("W").agg({
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum"
    }).dropna()

    if len(w) < 12:
        return False

    ma5 = w["c"].rolling(5).mean()
    ma10 = w["c"].rolling(10).mean()
    return bool((w["c"].iloc[-1] > ma5.iloc[-1]) and (ma5.iloc[-1] > ma10.iloc[-1]))


def grade(score):
    if score >= 95:
        return "S+급"
    if score >= 90:
        return "S급"
    if score >= 80:
        return "A급"
    if score >= 70:
        return "B급"
    if score >= 60:
        return "C급"
    return "제외"


def fmt_price(x):
    try:
        return f"{int(round(x, 0)):,}원"
    except Exception:
        return "-"


def build_signal_metrics(df, smooth=8):
    df = df.copy().dropna()
    ha = alpha_heikin_ashi(df, smooth=smooth)

    df["rsi"] = rsi(df["c"])
    df["adx"] = adx(df)
    df["ma20"] = df["c"].rolling(20).mean()
    df["vol_avg20"] = df["v"].rolling(20).mean()

    latest = df.iloc[-1]
    signal_date = df.index[-1].strftime("%Y-%m-%d")

    vol_ratio = latest["v"] / latest["vol_avg20"] if latest["vol_avg20"] and latest["vol_avg20"] > 0 else 0
    price = float(latest["c"])

    return {
        "signal_date": signal_date,
        "price": price,
        "rsi": float(latest["rsi"]) if pd.notna(latest["rsi"]) else 0,
        "adx": float(latest["adx"]) if pd.notna(latest["adx"]) else 0,
        "vol_ratio": float(vol_ratio),
        "above_ma20": bool(latest["c"] > latest["ma20"]) if pd.notna(latest["ma20"]) else False,
        "ha_bull_turn": is_ha_bull_turn(ha),
        "recent_doji": has_recent_doji(ha),
        "weekly_up": weekly_uptrend(df),
    }


def score_short_5(metrics):
    # 단기 5% 전략 / 총 100점
    score = 0
    reasons = []

    if metrics["ha_bull_turn"]:
        score += 30
        reasons.append("알파 하이킨아시 양전환 +30")

    if metrics["vol_ratio"] >= 1.3:
        add = min(20, int(15 + (metrics["vol_ratio"] - 1.3) * 10))
        score += add
        reasons.append(f"거래량 평균 대비 {metrics['vol_ratio']:.1f}배 +{add}")

    if metrics["rsi"] >= 50:
        add = min(20, int(10 + (metrics["rsi"] - 50) * 0.8))
        score += add
        reasons.append(f"RSI(상대강도지수) {metrics['rsi']:.1f} +{add}")

    if metrics["above_ma20"]:
        score += 20
        reasons.append("20일 이동평균선 위 +20")

    if metrics["recent_doji"]:
        score += 10
        reasons.append("최근 도지 후 전환 가능성 +10")

    return min(score, 100), reasons


def score_mid_30(metrics):
    # 중기 30% 전략 / 총 100점
    score = 0
    reasons = []

    if metrics["ha_bull_turn"]:
        score += 25
        reasons.append("알파 하이킨아시 양전환 +25")

    if metrics["adx"] >= 20:
        add = min(20, int(12 + (metrics["adx"] - 20) * 0.8))
        score += add
        reasons.append(f"ADX(추세강도지수) {metrics['adx']:.1f} +{add}")

    if metrics["vol_ratio"] >= 1.8:
        add = min(20, int(15 + (metrics["vol_ratio"] - 1.8) * 5))
        score += add
        reasons.append(f"거래량 평균 대비 {metrics['vol_ratio']:.1f}배 +{add}")

    if metrics["weekly_up"]:
        score += 20
        reasons.append("주봉 상승 추세 +20")

    if metrics["recent_doji"]:
        score += 15
        reasons.append("최근 도지 후 전환 +15")

    return min(score, 100), reasons


def make_trade_prices(price, strategy):
    if strategy == "short5":
        buy = price
        stop = price * 0.97
        target1 = price * 1.05
        risk = buy - stop
        reward = target1 - buy
        rr = reward / risk if risk > 0 else 0
        return {
            "buy": buy,
            "stop": stop,
            "target1": target1,
            "target2": None,
            "loss_pct": -3.0,
            "gain1_pct": 5.0,
            "gain2_pct": None,
            "rr": rr,
        }

    buy = price
    stop = price * 0.95
    target1 = price * 1.20
    target2 = price * 1.30
    risk = buy - stop
    reward = target2 - buy
    rr = reward / risk if risk > 0 else 0
    return {
        "buy": buy,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "loss_pct": -5.0,
        "gain1_pct": 20.0,
        "gain2_pct": 30.0,
        "rr": rr,
    }


def format_short_5(name, metrics, score, reasons):
    prices = make_trade_prices(metrics["price"], "short5")
    return f"""
⚡ [단기 5% 전략]

종목 : {name}
상태 : 🟢 신규신호
등급 : {grade(score)}
점수 : {score}점

신호일 : {metrics['signal_date']}
경과일 : 0일

매수가 : {fmt_price(prices['buy'])}
현재가 : {fmt_price(metrics['price'])}

손절가 : {fmt_price(prices['stop'])}
예상손실 : {prices['loss_pct']:.1f}%

목표가 : {fmt_price(prices['target1'])}
예상수익 : +{prices['gain1_pct']:.1f}%

손익비 : {prices['rr']:.1f} : 1

RSI (상대강도지수) : {metrics['rsi']:.1f}
ADX (추세강도지수) : {metrics['adx']:.1f}
거래량 : 평균 대비 {metrics['vol_ratio']:.1f}배
20일 이동평균선 : {'위' if metrics['above_ma20'] else '아래'}
알파 하이킨아시 : {'양전환' if metrics['ha_bull_turn'] else '미충족'}

점수상세
- """ + "\n- ".join(reasons)


def format_mid_30(name, metrics, score, reasons):
    prices = make_trade_prices(metrics["price"], "mid30")
    return f"""
📈 [중기 30% 전략]

종목 : {name}
상태 : 🟢 신규신호
등급 : {grade(score)}
점수 : {score}점

신호일 : {metrics['signal_date']}
경과일 : 0일

매수가 : {fmt_price(prices['buy'])}
현재가 : {fmt_price(metrics['price'])}

손절가 : {fmt_price(prices['stop'])}
예상손실 : {prices['loss_pct']:.1f}%

1차 목표가 : {fmt_price(prices['target1'])}
예상수익 : +{prices['gain1_pct']:.1f}%

2차 목표가 : {fmt_price(prices['target2'])}
예상수익 : +{prices['gain2_pct']:.1f}%

손익비 : {prices['rr']:.1f} : 1

RSI (상대강도지수) : {metrics['rsi']:.1f}
ADX (추세강도지수) : {metrics['adx']:.1f}
거래량 : 평균 대비 {metrics['vol_ratio']:.1f}배
주봉 추세 : {'상승' if metrics['weekly_up'] else '미충족'}
알파 하이킨아시 : {'양전환' if metrics['ha_bull_turn'] else '미충족'}

점수상세
- """ + "\n- ".join(reasons)


def analyze_one_stock(name, df, min_score=70):
    """
    기존 notify.py에서 종목별 df를 가져온 뒤 이 함수를 호출하면 됩니다.
    반환값: 텔레그램에 보낼 메시지 조각 리스트
    """
    metrics = build_signal_metrics(df)
    messages = []

    s5, r5 = score_short_5(metrics)
    if s5 >= min_score:
        messages.append(format_short_5(name, metrics, s5, r5))

    s30, r30 = score_mid_30(metrics)
    if s30 >= min_score:
        messages.append(format_mid_30(name, metrics, s30, r30))

    return messages
