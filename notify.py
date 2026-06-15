THIS_IS_NEW_VERSION = "0615"
print(THIS_IS_NEW_VERSION)
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

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


def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def alpha_heikin_ashi(df, smooth):
    o = ema(df["o"], smooth)
    h = ema(df["h"], smooth)
    l = ema(df["l"], smooth)
    c = ema(df["c"], smooth)

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
    high = df["h"]
    low = df["l"]
    close = df["c"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

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
            adx_value = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 0.0
            vr = float(vol_ratio.iloc[i])

            if body >= BODY_MIN and adx_value >= ADX_MIN and vr >= VOL_STRONG:
                return {
                    "signal_date": df.index[i].strftime("%Y-%m-%d"),
                    "days_ago": n - 1 - i,
                    "buy_price": float(df["c"].iloc[i]),
                    "stop_price": float(df["l"].iloc[i]),
                }

    return None


def get_tickers():
    try:
        import FinanceDataReader as fdr

        listing = fdr.StockListing("KOSPI")
        codes = listing.iloc[:200, 0].astype(str).str.zfill(6).tolist()

        if len(codes) >= 100:
            return codes
    except Exception as e:
        print("FinanceDataReader 실패:", e)

    try:
        from pykrx import stock

        today = datetime.today().strftime("%Y%m%d")
        cap = stock.get_market_cap(today, market="KOSPI")

        if cap is not None and not cap.empty:
            return cap.index[:200].tolist()
    except Exception as e:
        print("pykrx 시총 조회 실패:", e)

    return None


def scan():
    from pykrx import stock

    tickers = get_tickers()

    if not tickers:
        raise RuntimeError("종목 리스트 조회 실패")

    end = datetime.today()
    start = end - timedelta(days=FETCH_DAYS)

    s_str = start.strftime("%Y%m%d")
    e_str = end.strftime("%Y%m%d")

    hits = []

    for code in tickers:
        try:
            raw = stock.get_market_ohlcv(s_str, e_str, code)

            if raw is None or raw.empty:
                continue

            df = raw.rename(
                columns={
                    "시가": "o",
                    "고가": "h",
                    "저가": "l",
                    "종가": "c",
                    "거래량": "v",
                }
            )[["o", "h", "l", "c", "v"]].astype(float)

            df = df[df["c"] > 0]

            res = find_recent_up_flip(df)

            if res:
                try:
                    name = stock.get_market_ticker_name(code)
                except Exception:
                    name = code

                res["code"] = code
                res["name"] = name
                hits.append(res)

        except Exception as e:
            print(f"{code} 처리 실패:", e)

        time.sleep(SLEEP)

    if hits:
        dfh = pd.DataFrame(hits)
        dfh = dfh.sort_values(
            ["days_ago", "buy_price"],
            ascending=[True, False],
        ).reset_index(drop=True)
        hits = dfh.to_dict("records")

    return hits, end


def send_telegram(text):
    if not TOKEN or not CHAT:
        print("TELEGRAM_TOKEN 또는 TELEGRAM_CHAT 없음")
        return False

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT,
                "text": text,
            },
            timeout=20,
        )

        print("telegram status:", response.status_code)
        print("telegram response:", response.text)

        return response.status_code == 200

    except Exception as e:
        print("telegram error:", e)
        return False


def build_message(hits, end):
    date_str = end.strftime("%Y-%m-%d")

    if not hits:
        return f"kospi200\n{date_str}\n\nno strong signal"

    msg = f"kospi200 {date_str}\n강 신호 {len(hits)}개\n\n"

    for h in hits:
        msg += (
            f"{h['name']}({h['code']})\n"
            f"신호일 {h['signal_date']}\n"
            f"매수가격 {int(h['buy_price']):,}\n"
            f"손절가격 {int(h['stop_price']):,}\n\n"
        )

    return msg


def main():
    print("notify.py 매수가격 손절가격 버전 실행")

    try:
        hits, end = scan()
        msg = build_message(hits, end)
        send_telegram(msg)

    except Exception as e:
        error_msg = f"error: {e}"
        print(error_msg)
        send_telegram(error_msg)


if __name__ == "__main__":
    main()
