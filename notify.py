# notify.py
# VERSION: 0616_KOSPI200_FINAL
# GitHub Actions 전용
# 코스피200 CSV 기반 5% / 30% 전략 텔레그램 리포트
#
# 필요 파일:
# - notify.py
# - kospi200_tickers.csv
#
# 필요 패키지:
# pip install pandas numpy requests pykrx
#
# 필요 환경변수:
# TELEGRAM_TOKEN
# TELEGRAM_CHAT

THIS_IS_NEW_VERSION = "0616_KOSPI200_FINAL_TIGHT"
print("RUNNING VERSION:", THIS_IS_NEW_VERSION)

import os
import csv
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

try:
    from pykrx import stock
except Exception as e:
    stock = None
    print("pykrx import 실패:", e)


# =========================
# 기본 설정
# =========================

TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
CHAT = os.environ.get("TELEGRAM_CHAT", "").strip()

TICKER_FILE = "kospi200_tickers.csv"

FETCH_DAYS = 260
SLEEP = 0.08
SMOOTH = 8

# 후보 기준
MIN_SCORE_SHORT = 60      # 5% 전략
MIN_SCORE_MID = 70        # 30% 전략

MAX_SHORT_RESULTS = 5
MAX_MID_RESULTS = 3


# =========================
# 텔레그램
# =========================

def split_text(text, limit=3800):
    if len(text) <= limit:
        return [text]

    chunks = []
    current = ""

    for part in text.split("\n\n"):
        if len(current) + len(part) + 2 > limit:
            if current:
                chunks.append(current)
            current = part
        else:
            current += ("\n\n" if current else "") + part

    if current:
        chunks.append(current)

    return chunks


def send_telegram(text):
    if not TOKEN or not CHAT:
        print("텔레그램 TOKEN 또는 CHAT이 비어있습니다.")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    for chunk in split_text(text):
        try:
            r = requests.post(
                url,
                data={
                    "chat_id": CHAT,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if r.status_code != 200:
                print("텔레그램 발송 실패:", r.status_code, r.text)
            time.sleep(0.4)
        except Exception as e:
            print("텔레그램 오류:", e)


# =========================
# 지표 계산
# =========================

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(0)


def adx(df, period=14):
    high = df["h"]
    low = df["l"]
    close = df["c"]

    plus_dm_raw = high.diff()
    minus_dm_raw = -low.diff()

    plus_dm = np.where(
        (plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0),
        plus_dm_raw,
        0.0,
    )
    minus_dm = np.where(
        (minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0),
        minus_dm_raw,
        0.0,
    )

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = (
        100
        * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
        / atr.replace(0, np.nan)
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
        / atr.replace(0, np.nan)
    )

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


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

    rng = (out["ha_h"] - out["ha_l"]).replace(0, np.nan)

    out["ha_body_pct"] = ((out["ha_c"] - out["ha_o"]).abs() / rng) * 100

    out["lower_wick_pct"] = (
        (pd.concat([out["ha_o"], out["ha_c"]], axis=1).min(axis=1) - out["ha_l"]) / rng
    ) * 100

    return out.fillna(0)


def is_ha_bull_turn(ha):
    if len(ha) < 2:
        return False
    return bool(ha["ha_bear"].iloc[-2] and ha["ha_bull"].iloc[-1])


def has_recent_doji(ha, lookback=3, body_max_pct=18):
    recent = ha.tail(lookback)
    return bool((recent["ha_body_pct"] <= body_max_pct).any())


def strong_bull_ha(ha):
    if len(ha) < 1:
        return False
    last = ha.iloc[-1]
    return bool(
        last["ha_bull"]
        and last["ha_body_pct"] >= 30
        and last["lower_wick_pct"] <= 15
    )


def weekly_uptrend(df):
    w = df.resample("W").agg({
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }).dropna()

    if len(w) < 12:
        return False

    ma5 = w["c"].rolling(5).mean()
    ma10 = w["c"].rolling(10).mean()

    return bool(w["c"].iloc[-1] > ma5.iloc[-1] and ma5.iloc[-1] > ma10.iloc[-1])


# =========================
# 유틸
# =========================

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
    return "관찰"


def fmt_price(x):
    try:
        return f"{int(round(float(x), 0)):,}원"
    except Exception:
        return "-"


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# =========================
# 데이터
# =========================

def load_tickers():
    if not os.path.exists(TICKER_FILE):
        raise RuntimeError(f"{TICKER_FILE} 파일이 없습니다. GitHub 루트에 업로드하세요.")

    tickers = []

    with open(TICKER_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = str(row.get("code", "")).strip().zfill(6)
            name = str(row.get("name", "")).strip()

            if len(code) == 6 and code.isdigit() and name:
                tickers.append((code, name))

    print(f"KOSPI200 CSV 검색 대상: {len(tickers)}개")
    return tickers


def normalize_ohlcv(raw):
    if raw is None or raw.empty:
        return None

    df = raw.copy()

    rename_map = {
        "시가": "o",
        "고가": "h",
        "저가": "l",
        "종가": "c",
        "거래량": "v",
    }

    df = df.rename(columns=rename_map)

    required = ["o", "h", "l", "c", "v"]
    for col in required:
        if col not in df.columns:
            return None

    df = df[required].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < 60:
        return None

    return df


def get_ohlcv(code):
    if stock is None:
        return None

    end = datetime.now()
    start = end - timedelta(days=FETCH_DAYS)

    try:
        raw = stock.get_market_ohlcv_by_date(
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            code,
        )
        return normalize_ohlcv(raw)
    except Exception as e:
        print("OHLCV 조회 실패:", code, e)
        return None


# =========================
# 분석
# =========================

def build_metrics(df):
    df = df.copy().dropna()

    if len(df) < 60:
        return None

    ha = alpha_heikin_ashi(df, smooth=SMOOTH)

    df["rsi"] = rsi(df["c"])
    df["adx"] = adx(df)
    df["ma20"] = df["c"].rolling(20).mean()
    df["vol_avg20"] = df["v"].rolling(20).mean()

    latest = df.iloc[-1]

    vol_avg = safe_float(latest["vol_avg20"])
    vol_ratio = safe_float(latest["v"]) / vol_avg if vol_avg > 0 else 0

    return {
        "signal_date": df.index[-1].strftime("%Y-%m-%d"),
        "price": safe_float(latest["c"]),
        "rsi": safe_float(latest["rsi"]),
        "adx": safe_float(latest["adx"]),
        "vol_ratio": safe_float(vol_ratio),
        "above_ma20": bool(latest["c"] > latest["ma20"]) if pd.notna(latest["ma20"]) else False,
        "ha_bull_turn": is_ha_bull_turn(ha),
        "strong_bull_ha": strong_bull_ha(ha),
        "recent_doji": has_recent_doji(ha),
        "weekly_up": weekly_uptrend(df),
        "recent_low_10": safe_float(df["l"].tail(10).min()),
        "recent_low_20": safe_float(df["l"].tail(20).min()),
    }


def score_short_5(m):
    score = 0
    reasons = []

    if m["ha_bull_turn"]:
        score += 30
        reasons.append("알파 하이킨아시 양전환 +30")
    elif m["strong_bull_ha"]:
        score += 20
        reasons.append("알파 하이킨아시 강한 양봉 유지 +20")
    else:
        score += 10
        reasons.append("알파 하이킨아시 예비후보 +10")

    if m["vol_ratio"] >= 1.2:
        add = min(20, int(15 + (m["vol_ratio"] - 1.2) * 10))
        score += add
        reasons.append(f"거래량 평균 대비 {m['vol_ratio']:.1f}배 +{add}")

    if m["rsi"] >= 50:
        add = min(20, int(10 + (m["rsi"] - 50) * 0.8))
        score += add
        reasons.append(f"RSI(상대강도지수) {m['rsi']:.1f} +{add}")

    if m["above_ma20"]:
        score += 20
        reasons.append("20일 이동평균선 위 +20")

    if m["recent_doji"]:
        score += 10
        reasons.append("최근 도지 후 전환 가능성 +10")

    return min(score, 100), reasons


def score_mid_30(m):
    score = 0
    reasons = []

    if m["ha_bull_turn"]:
        score += 25
        reasons.append("알파 하이킨아시 양전환 +25")
    elif m["strong_bull_ha"]:
        score += 18
        reasons.append("알파 하이킨아시 강한 양봉 유지 +18")
    else:
        score += 8
        reasons.append("알파 하이킨아시 예비후보 +8")

    if m["adx"] >= 20:
        add = min(20, int(12 + (m["adx"] - 20) * 0.8))
        score += add
        reasons.append(f"ADX(추세강도지수) {m['adx']:.1f} +{add}")

    if m["vol_ratio"] >= 1.8:
        add = min(20, int(15 + (m["vol_ratio"] - 1.8) * 5))
        score += add
        reasons.append(f"거래량 평균 대비 {m['vol_ratio']:.1f}배 +{add}")

    if m["weekly_up"]:
        score += 20
        reasons.append("주봉 상승 추세 +20")

    if m["recent_doji"]:
        score += 15
        reasons.append("최근 도지 후 전환 +15")

    return min(score, 100), reasons


def make_prices(m, strategy):
    price = m["price"]

    if strategy == "short5":
        fixed_stop = price * 0.97
        chart_stop = m["recent_low_10"]

        if chart_stop > 0 and chart_stop > fixed_stop and chart_stop < price:
            stop = chart_stop
            stop_type = "최근 10일 저점 기준"
        else:
            stop = fixed_stop
            stop_type = "고정 -3% 기준"

        target1 = price * 1.05
        risk = price - stop
        reward = target1 - price
        rr = reward / risk if risk > 0 else 0

        return stop, stop_type, target1, None, ((stop / price) - 1) * 100, rr

    fixed_stop = price * 0.95
    chart_stop = m["recent_low_20"]

    if chart_stop > 0 and chart_stop > fixed_stop and chart_stop < price:
        stop = chart_stop
        stop_type = "최근 20일 저점 기준"
    else:
        stop = fixed_stop
        stop_type = "고정 -5% 기준"

    target1 = price * 1.20
    target2 = price * 1.30
    risk = price - stop
    reward = target2 - price
    rr = reward / risk if risk > 0 else 0

    return stop, stop_type, target1, target2, ((stop / price) - 1) * 100, rr


def format_short(name, code, m, score, reasons):
    stop, stop_type, target1, _, loss_pct, rr = make_prices(m, "short5")

    return f"""⚡ [단기 5% 전략]

종목 : {name} ({code})
등급 : {grade(score)}
점수 : {score}점

신호일 : {m['signal_date']}
경과일 : 0일

매수가 : {fmt_price(m['price'])}
현재가 : {fmt_price(m['price'])}

손절가 : {fmt_price(stop)}
손절기준 : {stop_type}
예상손실 : {loss_pct:.1f}%

목표가 : {fmt_price(target1)}
예상수익 : +5.0%

손익비 : {rr:.1f} : 1

RSI (상대강도지수) : {m['rsi']:.1f}
ADX (추세강도지수) : {m['adx']:.1f}
거래량 : 평균 대비 {m['vol_ratio']:.1f}배
20일 이동평균선 : {'위' if m['above_ma20'] else '아래'}
알파 하이킨아시 : {'양전환' if m['ha_bull_turn'] else '강한 양봉 유지' if m['strong_bull_ha'] else '예비'}

점수상세
- """ + "\n- ".join(reasons)


def format_mid(name, code, m, score, reasons):
    stop, stop_type, target1, target2, loss_pct, rr = make_prices(m, "mid30")

    return f"""📈 [중기 30% 전략]

종목 : {name} ({code})
등급 : {grade(score)}
점수 : {score}점

신호일 : {m['signal_date']}
경과일 : 0일

매수가 : {fmt_price(m['price'])}
현재가 : {fmt_price(m['price'])}

손절가 : {fmt_price(stop)}
손절기준 : {stop_type}
예상손실 : {loss_pct:.1f}%

1차 목표가 : {fmt_price(target1)}
예상수익 : +20.0%

2차 목표가 : {fmt_price(target2)}
예상수익 : +30.0%

손익비 : {rr:.1f} : 1

RSI (상대강도지수) : {m['rsi']:.1f}
ADX (추세강도지수) : {m['adx']:.1f}
거래량 : 평균 대비 {m['vol_ratio']:.1f}배
주봉 추세 : {'상승' if m['weekly_up'] else '미충족'}
알파 하이킨아시 : {'양전환' if m['ha_bull_turn'] else '강한 양봉 유지' if m['strong_bull_ha'] else '예비'}

점수상세
- """ + "\n- ".join(reasons)


# =========================
# 메인
# =========================

def main():
    start_time = datetime.now()
    today = start_time.strftime("%Y-%m-%d")

    if stock is None:
        send_telegram("pykrx가 설치되지 않았습니다.")
        return

    try:
        tickers = load_tickers()
    except Exception as e:
        send_telegram(f"종목 파일 오류: {e}")
        return

    short_results = []
    mid_results = []

    checked_count = 0
    data_ok_count = 0
    metrics_ok_count = 0
    error_count = 0

    for i, (code, name) in enumerate(tickers, 1):
        try:
            checked_count += 1

            df = get_ohlcv(code)

            if df is None:
                continue

            data_ok_count += 1

            m = build_metrics(df)
            if m is None:
                continue

            metrics_ok_count += 1

            s5, r5 = score_short_5(m)
            s30, r30 = score_mid_30(m)

            if s5 >= MIN_SCORE_SHORT:
                short_results.append({
                    "score": s5,
                    "message": format_short(name, code, m, s5, r5),
                })

            if s30 >= MIN_SCORE_MID:
                mid_results.append({
                    "score": s30,
                    "message": format_mid(name, code, m, s30, r30),
                })

            if i % 50 == 0:
                print(f"진행: {i}/{len(tickers)} / 데이터 {data_ok_count} / 지표 {metrics_ok_count}")

            time.sleep(SLEEP)

        except Exception as e:
            error_count += 1
            print("분석 오류:", code, name, e)

    short_results = sorted(short_results, key=lambda x: x["score"], reverse=True)[:MAX_SHORT_RESULTS]
    mid_results = sorted(mid_results, key=lambda x: x["score"], reverse=True)[:MAX_MID_RESULTS]

    elapsed = datetime.now() - start_time

    header = f"""📊 장마감 전략 리포트

기준일 : {today}
버전 : {THIS_IS_NEW_VERSION}

⚡ 단기 5% 전략 후보 : {len(short_results)}개
📈 중기 30% 전략 후보 : {len(mid_results)}개

검색대상 : {len(tickers)}개
시도종목 : {checked_count}개
데이터성공 : {data_ok_count}개
지표성공 : {metrics_ok_count}개
오류종목 : {error_count}개
소요시간 : {str(elapsed).split('.')[0]}
"""

    parts = [header]

    if short_results:
        parts.append("━━━━━━━━━━━━━━\n⚡ 단기 5% 전략 TOP\n━━━━━━━━━━━━━━")
        for idx, r in enumerate(short_results, 1):
            parts.append(f"[{idx}]\n{r['message']}")
    else:
        parts.append("⚡ 단기 5% 전략\n후보 없음")

    if mid_results:
        parts.append("━━━━━━━━━━━━━━\n📈 중기 30% 전략 TOP\n━━━━━━━━━━━━━━")
        for idx, r in enumerate(mid_results, 1):
            parts.append(f"[{idx}]\n{r['message']}")
    else:
        parts.append("📈 중기 30% 전략\n후보 없음")

    text = "\n\n".join(parts)
    send_telegram(text)

    print("완료")
    print(header)


if __name__ == "__main__":
    main()
