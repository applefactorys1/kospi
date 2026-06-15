# -*- coding: utf-8 -*-
"""
코스피200 상승전환 스크리너 → 텔레그램 자동 발송
알파 하이킨아시(평활 강함) + 추세 강도 필터

매일 정해진 시간에 자동 실행되어, 조건을 통과한 상승 전환 종목을
텔레그램 봇으로 보내줍니다. (GitHub Actions로 24시간 무인 구동)

[필요 환경변수]
  TELEGRAM_TOKEN : BotFather가 준 봇 토큰
  TELEGRAM_CHAT  : 내 chat_id (@userinfobot 으로 확인)

[로컬 테스트]
  export TELEGRAM_TOKEN="123456:ABC..."   (Windows: set TELEGRAM_TOKEN=...)
  export TELEGRAM_CHAT="123456789"
  python notify.py
"""

import os
import time
from datetime import datetime, timedelta

import requests
import pandas as pd

# ===== 조건 설정 =====
SMOOTH = 8
BODY_MIN = 35.0
ADX_MIN = 22.0
RECENT_DAYS = 10
VOL_AVG = 20
VOL_STRONG = 1.8
FETCH_DAYS = 150
SLEEP = 0.12

TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
CHAT = os.environ.get("TELEGRAM_CHAT", "").strip()


# ---------- 지표 ----------
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def alpha_heikin_ashi(df, smooth):
    o = ema(df["o"], smooth); h = ema(df["h"], smooth)
    l = ema(df["l"], smooth); c = ema(df["c"], smooth)
    ha_close = (o + h + l + c) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)
    out = pd.DataFrame(index=df.index)
    out["up"] = ha_close >= ha_open
    rng = (ha_high - ha_low).replace(0, 1e-9)
    out["body"] = (ha_close - ha_open).abs() / rng * 100
    return out


def compute_adx(df, period=14):
    high, low, close = df["h"], df["l"], df["c"]
    up_move = high.diff(); down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-9)
    ndi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-9)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, 1e-9)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def find_recent_up_flip(df):
    if len(df) < 40:
        return None
    ha = alpha_heikin_ashi(df, SMOOTH)
    adx = compute_adx(df, 14)
    vol_ma = df["v"].rolling(VOL_AVG, min_periods=5).mean().shift(1)
    vol_ratio = (df["v"] / vol_ma).fillna(1.0)
    ups = ha["up"].values
    n = len(df)
    for i in range(n - 1, max(0, n - 1 - RECENT_DAYS), -1):
        if i == 0:
            break
        if ups[i] and not ups[i - 1]:
            body = ha["body"].iloc[i]
            a = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 0.0
            if body >= BODY_MIN and a >= ADX_MIN:
                vr = float(vol_ratio.iloc[i])
                return {
                    "flip_date": df.index[i].strftime("%m-%d"),
                    "days_ago": n - 1 - i,
                    "close": float(df["c"].iloc[-1]),
                    "body": round(body, 0),
                    "adx": round(a, 0),
                    "vol_ratio": round(vr, 1),
                    "grade": "강" if vr >= VOL_STRONG else "중",
                }
    return None


def get_tickers_and_names():
    name_map = {}
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KOSPI")
        code_col = next((c for c in ["Code", "Symbol"] if c in df.columns), None)
        cap_col = next((c for c in ["Marcap", "MarketCap", "시가총액"] if c in df.columns), None)
        name_col = next((c for c in ["Name", "종목명"] if c in df.columns), None)
        if code_col:
            df = df.dropna(subset=[code_col])
            df = df[df[code_col].astype(str).str.match(r"^\d{6}$")]
            if cap_col and cap_col in df.columns:
                df = df.sort_values(cap_col, ascending=False)
            top = df.head(200)
            codes = top[code_col].astype(str).str.zfill(6).tolist()
            if name_col:
                for _, r in top.iterrows():
                    name_map[str(r[code_col]).zfill(6)] = str(r[name_col])
            if len(codes) >= 100:
                return codes, name_map
    except Exception:
        pass
    from pykrx import stock
    today = datetime.today()
    for back in range(0, 10):
        d = (today - timedelta(days=back)).strftime("%Y%m%d")
        try:
            cap = stock.get_market_cap(d, market="KOSPI")
            if cap is not None and not cap.empty:
                top = cap.sort_values("시가총액", ascending=False).head(200).index.tolist()
                if len(top) >= 100:
                    return top, name_map
        except Exception:
            continue
    raise RuntimeError("종목 목록 조회 실패")


def scan():
    from pykrx import stock
    tickers, name_map = get_tickers_and_names()
    end = datetime.today()
    start = end - timedelta(days=FETCH_DAYS)
    s_str, e_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    hits = []
    for code in tickers:
        try:
            raw = stock.get_market_ohlcv(s_str, e_str, code)
            if raw is None or raw.empty:
                continue
            df = raw.rename(columns={"시가": "o", "고가": "h", "저가": "l",
                                     "종가": "c", "거래량": "v"})[["o", "h", "l", "c", "v"]].astype(float)
            df = df[df["c"] > 0]
            res = find_recent_up_flip(df)
            if res:
                nm = name_map.get(code)
                if not nm:
                    try:
                        nm = stock.get_market_ticker_name(code)
                    except Exception:
                        nm = code
                res.update({"code": code, "name": nm})
                hits.append(res)
        except Exception:
            pass
        time.sleep(SLEEP)

    if hits:
        dfh = pd.DataFrame(hits)
        dfh["gr"] = dfh["grade"].map({"강": 0, "중": 1})
        dfh = dfh.sort_values(["gr", "days_ago", "vol_ratio"],
                              ascending=[True, True, False]).reset_index(drop=True)
        hits = dfh.drop(columns=["gr"]).to_dict("records")
    return hits, end


def build_message(hits, end):
    date_str = end.strftime("%Y-%m-%d")
    if not hits:
        return (f"📊 *코스피200 상승전환 스크리너*\n"
                f"_{date_str} 기준_\n\n"
                f"오늘은 조건(평활 EMA{SMOOTH} · 몸통≥{int(BODY_MIN)}% · ADX≥{int(ADX_MIN)} · "
                f"최근 {RECENT_DAYS}일)을 통과한 상승 전환 종목이 없습니다.\n"
                f"엄격 조건에서는 종목이 드물게 나오는 게 정상입니다.")

    strong = sum(1 for h in hits if h["grade"] == "강")
    lines = [f"📊 *코스피200 상승전환 스크리너*",
             f"_{date_str} 기준 · {len(hits)}종목 (강★ {strong})_", ""]
    for h in hits:
        star = "🔥" if h["grade"] == "강" else "•"
        lines.append(
            f"{star} *{h['name']}* ({h['code']})\n"
            f"   전환 {h['flip_date']}({h['days_ago']}일전) · "
            f"{int(h['close']):,}원 · 몸통{int(h['body'])}% ADX{int(h['adx'])} · 거래량 {h['vol_ratio']}배"
        )
    lines.append("\n_참고용입니다. 투자 책임은 본인에게 있습니다._")
    return "\n".join(lines)


def send_telegram(text):
    if not TOKEN or not CHAT:
        print("환경변수 TELEGRAM_TOKEN / TELEGRAM_CHAT 가 없습니다.")
        print("---- 메시지 미리보기 ----")
        print(text)
        return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    # 텔레그램 메시지 길이 제한(4096자) 대비 분할
    chunks = [text[i:i + 3500] for i in range(0, len(text), 3500)] or [text]
    ok = True
    for ch in chunks:
        try:
            r = requests.post(url, data={"chat_id": CHAT, "text": ch,
                                         "parse_mode": "Markdown"}, timeout=20)
            if r.status_code != 200:
                print("텔레그램 발송 실패:", r.status_code, r.text)
                ok = False
        except Exception as e:
            print("발송 오류:", e)
            ok = False
    return ok


def main():
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] 스캔 시작")
    try:
        hits, end = scan()
    except Exception as e:
        send_telegram(f"⚠️ 스크리너 실행 중 오류: {e}")
        print("오류:", e)
        return
    msg = build_message(hits, end)
    sent = send_telegram(msg)
    print("발송 완료" if sent else "발송 안 됨(미리보기만 출력)")


if __name__ == "__main__":
    main()
