# notify.py
# VERSION: 0616_DUAL_STRATEGY_SCORE
# 매일 오후 6시 실행용: 5% 전략 + 30% 전략 텔레그램 종목 리포트
#
# 필요 패키지:
# pip install pandas numpy requests pykrx
#
# 환경변수 필요:
# TELEGRAM_TOKEN
# TELEGRAM_CHAT
#
# 실행:
# python notify.py

THIS_IS_NEW_VERSION = "0616_DUAL_STRATEGY_SCORE_TOP_ALWAYS"
print("RUNNING VERSION:", THIS_IS_NEW_VERSION)

import os
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

FETCH_DAYS = 260          # 일봉 데이터 확보 기간
SLEEP = 0.08              # 종목 조회 간격
SMOOTH = 8                # 알파 하이킨아시 smoothing
MIN_SCORE_SHORT = 50      # 5% 전략: 후보 확인용으로 완화
MIN_SCORE_MID = 60        # 30% 전략: 후보 확인용으로 완화
MAX_SHORT_RESULTS = 20    # 5% 전략 최대 표시
MAX_MID_RESULTS = 15       # 30% 전략 최대 표시

# 전체 시장 검색 여부
SCAN_KOSPI = True
SCAN_KOSDAQ = True

# tickers.txt가 있으면 그 종목만 검색
# 예시:
# 005930,삼성전자
# 042700,한미반도체
TICKER_FILE = "tickers.txt"


# =========================
# 텔레그램
# =========================

def send_telegram(text):
    if not TOKEN or not CHAT:
        print("텔레그램 TOKEN 또는 CHAT이 비어있습니다.")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    # 텔레그램 메시지 길이 제한 대응
    chunks = split_text(text, 3800)

    for chunk in chunks:
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
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)


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
    out["ha_body_pct"] = (
        (out["ha_c"] - out["ha_o"]).abs()
        / (out["ha_h"] - out["ha_l"]).replace(0, np.nan)
    ) * 100

    out["lower_wick_pct"] = (
        (pd.concat([out["ha_o"], out["ha_c"]], axis=1).min(axis=1) - out["ha_l"])
        / (out["ha_h"] - out["ha_l"]).replace(0, np.nan)
    ) * 100

    out["upper_wick_pct"] = (
        (out["ha_h"] - pd.concat([out["ha_o"], out["ha_c"]], axis=1).max(axis=1))
        / (out["ha_h"] - out["ha_l"]).replace(0, np.nan)
    ) * 100

    return out.fillna(0)


def is_ha_bull_turn(ha):
    if len(ha) < 2:
        return False
    return bool(ha["ha_bear"].iloc[-2] and ha["ha_bull"].iloc[-1])


def is_ha_bear_turn(ha):
    if len(ha) < 2:
        return False
    return bool(ha["ha_bull"].iloc[-2] and ha["ha_bear"].iloc[-1])


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
    # 주봉 종가가 5주선 위 + 5주선이 10주선 위면 상승 추세로 판단
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

    return bool((w["c"].iloc[-1] > ma5.iloc[-1]) and (ma5.iloc[-1] > ma10.iloc[-1]))


# =========================
# 점수 계산
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
    return "제외"


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


def build_signal_metrics(df):
    df = df.copy().dropna()

    if len(df) < 60:
        return None

    ha = alpha_heikin_ashi(df, smooth=SMOOTH)

    df["rsi"] = rsi(df["c"])
    df["adx"] = adx(df)
    df["ma20"] = df["c"].rolling(20).mean()
    df["vol_avg20"] = df["v"].rolling(20).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signal_date = df.index[-1].strftime("%Y-%m-%d")
    price = safe_float(latest["c"])

    vol_avg = safe_float(latest["vol_avg20"])
    vol_ratio = safe_float(latest["v"]) / vol_avg if vol_avg > 0 else 0

    # 최근 저점 기반 보조 손절가
    recent_low_10 = safe_float(df["l"].tail(10).min())
    recent_low_20 = safe_float(df["l"].tail(20).min())

    return {
        "signal_date": signal_date,
        "price": price,
        "rsi": safe_float(latest["rsi"]),
        "adx": safe_float(latest["adx"]),
        "vol_ratio": safe_float(vol_ratio),
        "above_ma20": bool(latest["c"] > latest["ma20"]) if pd.notna(latest["ma20"]) else False,
        "ha_bull_turn": is_ha_bull_turn(ha),
        "ha_bear_turn": is_ha_bear_turn(ha),
        "strong_bull_ha": strong_bull_ha(ha),
        "recent_doji": has_recent_doji(ha),
        "weekly_up": weekly_uptrend(df),
        "recent_low_10": recent_low_10,
        "recent_low_20": recent_low_20,
    }


def score_short_5(metrics):
    # 단기 5% 전략 / 총 100점
    score = 0
    reasons = []

    if metrics["ha_bull_turn"]:
        score += 30
        reasons.append("알파 하이킨아시 양전환 +30")
    elif metrics["strong_bull_ha"]:
        score += 20
        reasons.append("알파 하이킨아시 강한 양봉 유지 +20")
    else:
        # 양전환이 오늘 딱 나오지 않아도, 다른 조건이 좋으면 후보로 보기 위해 기본점수 부여
        score += 10
        reasons.append("알파 하이킨아시 양전환 전 예비후보 +10")

    if metrics["vol_ratio"] >= 1.2:
        add = min(20, int(15 + (metrics["vol_ratio"] - 1.2) * 10))
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
    elif metrics["strong_bull_ha"]:
        score += 18
        reasons.append("알파 하이킨아시 강한 양봉 유지 +18")
    else:
        # 30% 전략도 주봉/ADX/거래량이 좋으면 예비후보로 확인
        score += 8
        reasons.append("알파 하이킨아시 양전환 전 예비후보 +8")

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


# =========================
# 매수가 / 목표가 / 손절가
# =========================

def make_trade_prices(metrics, strategy):
    price = metrics["price"]

    if strategy == "short5":
        buy = price
        fixed_stop = price * 0.97
        chart_stop = metrics["recent_low_10"]

        # 5% 전략은 -3% 기준을 우선. 단, 최근 저점이 더 가까우면 최근 저점 사용
        if chart_stop > 0 and chart_stop > fixed_stop and chart_stop < buy:
            stop = chart_stop
            stop_type = "최근 10일 저점 기준"
        else:
            stop = fixed_stop
            stop_type = "고정 -3% 기준"

        target1 = price * 1.05
        risk = buy - stop
        reward = target1 - buy
        rr = reward / risk if risk > 0 else 0
        loss_pct = ((stop / buy) - 1) * 100 if buy > 0 else 0

        return {
            "buy": buy,
            "stop": stop,
            "stop_type": stop_type,
            "target1": target1,
            "target2": None,
            "loss_pct": loss_pct,
            "gain1_pct": 5.0,
            "gain2_pct": None,
            "rr": rr,
        }

    buy = price
    fixed_stop = price * 0.95
    chart_stop = metrics["recent_low_20"]

    # 30% 전략은 -5% 기준을 우선. 단, 최근 20일 저점이 더 가까우면 최근 저점 사용
    if chart_stop > 0 and chart_stop > fixed_stop and chart_stop < buy:
        stop = chart_stop
        stop_type = "최근 20일 저점 기준"
    else:
        stop = fixed_stop
        stop_type = "고정 -5% 기준"

    target1 = price * 1.20
    target2 = price * 1.30
    risk = buy - stop
    reward = target2 - buy
    rr = reward / risk if risk > 0 else 0
    loss_pct = ((stop / buy) - 1) * 100 if buy > 0 else 0

    return {
        "buy": buy,
        "stop": stop,
        "stop_type": stop_type,
        "target1": target1,
        "target2": target2,
        "loss_pct": loss_pct,
        "gain1_pct": 20.0,
        "gain2_pct": 30.0,
        "rr": rr,
    }


# =========================
# 메시지 포맷
# =========================

def format_short_5(name, code, metrics, score, reasons):
    prices = make_trade_prices(metrics, "short5")

    return f"""⚡ [단기 5% 전략]

종목 : {name} ({code})
상태 : 🟢 신규신호
등급 : {grade(score)}
점수 : {score}점

신호일 : {metrics['signal_date']}
경과일 : 0일

매수가 : {fmt_price(prices['buy'])}
현재가 : {fmt_price(metrics['price'])}

손절가 : {fmt_price(prices['stop'])}
손절기준 : {prices['stop_type']}
예상손실 : {prices['loss_pct']:.1f}%

목표가 : {fmt_price(prices['target1'])}
예상수익 : +{prices['gain1_pct']:.1f}%

손익비 : {prices['rr']:.1f} : 1

RSI (상대강도지수) : {metrics['rsi']:.1f}
→ 50 이상이면 매수세 우위

ADX (추세강도지수) : {metrics['adx']:.1f}
→ 20 이상이면 추세 시작

거래량 : 평균 대비 {metrics['vol_ratio']:.1f}배
20일 이동평균선 : {'위' if metrics['above_ma20'] else '아래'}
알파 하이킨아시 : {'양전환' if metrics['ha_bull_turn'] else '강한 양봉 유지' if metrics['strong_bull_ha'] else '미충족'}

점수상세
- """ + "\n- ".join(reasons)


def format_mid_30(name, code, metrics, score, reasons):
    prices = make_trade_prices(metrics, "mid30")

    return f"""📈 [중기 30% 전략]

종목 : {name} ({code})
상태 : 🟢 신규신호
등급 : {grade(score)}
점수 : {score}점

신호일 : {metrics['signal_date']}
경과일 : 0일

매수가 : {fmt_price(prices['buy'])}
현재가 : {fmt_price(metrics['price'])}

손절가 : {fmt_price(prices['stop'])}
손절기준 : {prices['stop_type']}
예상손실 : {prices['loss_pct']:.1f}%

1차 목표가 : {fmt_price(prices['target1'])}
예상수익 : +{prices['gain1_pct']:.1f}%

2차 목표가 : {fmt_price(prices['target2'])}
예상수익 : +{prices['gain2_pct']:.1f}%

손익비 : {prices['rr']:.1f} : 1

RSI (상대강도지수) : {metrics['rsi']:.1f}
→ 50 이상이면 매수세 우위

ADX (추세강도지수) : {metrics['adx']:.1f}
→ 20 이상이면 추세 시작, 25 이상이면 강한 추세

거래량 : 평균 대비 {metrics['vol_ratio']:.1f}배
주봉 추세 : {'상승' if metrics['weekly_up'] else '미충족'}
알파 하이킨아시 : {'양전환' if metrics['ha_bull_turn'] else '강한 양봉 유지' if metrics['strong_bull_ha'] else '미충족'}

점수상세
- """ + "\n- ".join(reasons)


# =========================
# 데이터 수집
# =========================

def normalize_ohlcv(raw):
    if raw is None or raw.empty:
        return None

    df = raw.copy()

    # pykrx 컬럼명 대응
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

    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    try:
        raw = stock.get_market_ohlcv_by_date(start_s, end_s, code)
        return normalize_ohlcv(raw)
    except Exception as e:
        print("OHLCV 조회 실패:", code, e)
        return None


def load_tickers_from_file():
    path = TICKER_FILE
    if not os.path.exists(path):
        return []

    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                code, name = line.split(",", 1)
                tickers.append((code.strip().zfill(6), name.strip()))
            else:
                code = line.strip().zfill(6)
                tickers.append((code, code))

    return tickers


def get_market_tickers():
    file_tickers = load_tickers_from_file()
    if file_tickers:
        print(f"tickers.txt 기준 검색: {len(file_tickers)}개")
        return file_tickers

    if stock is None:
        raise RuntimeError("pykrx가 필요합니다. pip install pykrx 실행하세요.")

    tickers = []

    today = datetime.now().strftime("%Y%m%d")

    if SCAN_KOSPI:
        for code in stock.get_market_ticker_list(today, market="KOSPI"):
            name = stock.get_market_ticker_name(code)
            tickers.append((code, name))

    if SCAN_KOSDAQ:
        for code in stock.get_market_ticker_list(today, market="KOSDAQ"):
            name = stock.get_market_ticker_name(code)
            tickers.append((code, name))

    # ETF/ETN/스팩 등 일부 이름 필터
    exclude_words = ["스팩", "SPAC", "ETN", "KODEX", "TIGER", "ACE", "SOL", "KOSEF", "KBSTAR", "HANARO", "ARIRANG"]
    filtered = []
    for code, name in tickers:
        if any(w in name.upper() for w in exclude_words):
            continue
        filtered.append((code, name))

    print(f"시장 전체 검색 대상: {len(filtered)}개")
    return filtered


# =========================
# 종목 분석
# =========================

def analyze_one_stock(code, name):
    df = get_ohlcv(code)
    if df is None:
        return None

    metrics = build_signal_metrics(df)
    if metrics is None:
        return None

    short_score, short_reasons = score_short_5(metrics)
    mid_score, mid_reasons = score_mid_30(metrics)

    result = {
        "code": code,
        "name": name,
        "metrics": metrics,
        "short_score": short_score,
        "short_reasons": short_reasons,
        "mid_score": mid_score,
        "mid_reasons": mid_reasons,
        "short_message": None,
        "mid_message": None,
    }

    if short_score >= MIN_SCORE_SHORT:
        result["short_message"] = format_short_5(name, code, metrics, short_score, short_reasons)

    if mid_score >= MIN_SCORE_MID:
        result["mid_message"] = format_mid_30(name, code, metrics, mid_score, mid_reasons)

    return result


# =========================
# 메인
# =========================

def main():
    start_time = datetime.now()
    today = start_time.strftime("%Y-%m-%d")

    if stock is None:
        send_telegram("pykrx가 설치되지 않았습니다.\n\npip install pykrx 실행 후 다시 실행하세요.")
        return

    tickers = get_market_tickers()

    short_results = []
    mid_results = []

    total = len(tickers)

    for i, (code, name) in enumerate(tickers, 1):
        try:
            result = analyze_one_stock(code, name)

            if result:
                if result["short_message"]:
                    short_results.append(result)
                if result["mid_message"]:
                    mid_results.append(result)

            if i % 100 == 0:
                print(f"진행: {i}/{total} / 5% 후보 {len(short_results)} / 30% 후보 {len(mid_results)}")

            time.sleep(SLEEP)

        except Exception as e:
            print("분석 오류:", code, name, e)

    # 점수순 정렬
    short_results = sorted(short_results, key=lambda x: x["short_score"], reverse=True)[:MAX_SHORT_RESULTS]
    mid_results = sorted(mid_results, key=lambda x: x["mid_score"], reverse=True)[:MAX_MID_RESULTS]

    elapsed = datetime.now() - start_time

    header = f"""📊 장마감 전략 리포트

기준일 : {today}
버전 : {THIS_IS_NEW_VERSION}

⚡ 단기 5% 전략 후보 : {len(short_results)}개
📈 중기 30% 전략 후보 : {len(mid_results)}개

검색대상 : {total}개
소요시간 : {str(elapsed).split('.')[0]}
"""

    parts = [header]

    if short_results:
        parts.append("━━━━━━━━━━━━━━\n⚡ 단기 5% 전략 TOP\n━━━━━━━━━━━━━━")
        for idx, r in enumerate(short_results, 1):
            parts.append(f"\n[{idx}]\n" + r["short_message"])
    else:
        parts.append("⚡ 단기 5% 전략\n금일 B급 이상 후보 없음")

    if mid_results:
        parts.append("━━━━━━━━━━━━━━\n📈 중기 30% 전략 TOP\n━━━━━━━━━━━━━━")
        for idx, r in enumerate(mid_results, 1):
            parts.append(f"\n[{idx}]\n" + r["mid_message"])
    else:
        parts.append("📈 중기 30% 전략\n금일 B급 이상 후보 없음")

    text = "\n\n".join(parts)
    send_telegram(text)

    print("완료")
    print(header)


if __name__ == "__main__":
    main()
