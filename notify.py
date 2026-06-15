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


def build_html(hits, end):
    """HTML 결과 페이지 생성"""
    date_str = end.strftime("%Y-%m-%d")
    strong = sum(1 for h in hits if h["grade"] == "강") if hits else 0
    
    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>코스피200 상승전환 스크리너 - {date_str}</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--line:#283041;--ink:#e6edf3;--muted:#8b97a7;--gold:#d4a23a;--up:#22a06b;--down:#d6453d;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:radial-gradient(800px 460px at 100% -10%,rgba(212,162,58,.07),transparent 60%),var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;min-height:100vh;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:960px;margin:0 auto;padding:24px 16px 60px}}
.eyebrow{{font-size:11px;letter-spacing:.24em;text-transform:uppercase;color:var(--gold);font-weight:700;margin-bottom:9px}}
h1{{font-size:25px;font-weight:800;letter-spacing:-.02em}}
h1 .sub{{display:block;color:var(--muted);font-size:.52em;font-weight:600;margin-top:9px;line-height:1.45}}
.meta{{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 18px}}
.chip{{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:6px 13px;font-size:12px;color:var(--muted)}}
.chip b{{color:var(--ink);font-weight:700}}
.summary{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
.stat{{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:15px 18px;flex:1;min-width:120px}}
.stat .n{{font-size:28px;font-weight:800}} .stat .l{{font-size:12px;color:var(--muted);margin-top:3px}}
.stat.s .n{{color:var(--gold)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:13px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:15px;position:relative;overflow:hidden}}
.card::before{{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--up)}}
.card .top{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px}}
.card .name{{font-size:15px;font-weight:800}} .card .code{{font-size:12px;color:var(--muted);margin-top:2px}}
.badge{{font-size:11px;font-weight:800;padding:4px 10px;border-radius:999px;white-space:nowrap}}
.badge.s{{background:rgba(212,162,58,.18);color:var(--gold)}} .badge.m{{background:rgba(139,151,167,.15);color:var(--muted)}}
canvas{{width:100%;height:60px;display:block;margin:6px 0 10px}}
.row{{display:flex;justify-content:space-between;font-size:12.5px;padding:3px 0;color:var(--muted)}}
.row b{{color:var(--ink);font-weight:700;font-variant-numeric:tabular-nums}}
.empty{{text-align:center;color:var(--muted);padding:40px 20px;background:var(--panel);border:1px solid var(--line);border-radius:15px;line-height:1.7}}
footer{{color:var(--muted);font-size:11px;line-height:1.6;border-top:1px solid var(--line);padding-top:16px;margin-top:30px}}
</style></head><body><div class="wrap">
<div class="eyebrow">KOSPI200 · Alpha Heikin-Ashi Screener</div>
<h1>상승 전환 포착<span class="sub">평활 하이킨아시 색 전환 + 추세 강도 필터를 통과한 종목을 찾습니다</span></h1>
<div class="meta">
<span class="chip">기준일 <b>{date_str}</b></span>
<span class="chip">대상 <b>200종목</b></span>
<span class="chip">평활 <b>EMA{SMOOTH}</b></span>
<span class="chip">몸통 <b>≥{int(BODY_MIN)}%</b></span>
<span class="chip">ADX <b>≥{int(ADX_MIN)}</b></span>
<span class="chip">최근 <b>{RECENT_DAYS}일</b></span>
</div>
<div class="summary">
<div class="stat"><div class="n">{len(hits)}</div><div class="l">발견 종목</div></div>
<div class="stat s"><div class="n">{strong}</div><div class="l">강★ (거래량 급증)</div></div>
<div class="stat"><div class="n">{len(hits)-strong}</div><div class="l">중 (추세)</div></div>
</div>
"""
    
    if not hits:
        html += f'<div class="empty">조건을 만족하는 상승 전환 종목이 없습니다.<br>평활 강함 조건에서는 종목이 드물게 나오는 게 정상입니다.</div>'
    else:
        html += '<div class="grid">'
        for idx, h in enumerate(hits):
            isS = h["grade"] == "강"
            spark = h.get("spark", [])
            flip_pos = h.get("flip_pos", -1)
            html += f'''<div class="card">
<div class="top"><div><div class="name">{h['name']}</div><div class="code">{h['code']}</div></div>
<span class="badge {'s' if isS else 'm'}">{'강 ★' if isS else '중'}</span></div>
<canvas id="sp{idx}"></canvas>
<div class="row"><span>전환일</span><b>{h['flip_date']} ({h['days_ago']}일전)</b></div>
<div class="row"><span>현재가</span><b>{int(h['close']):,}원</b></div>
<div class="row"><span>몸통 / ADX</span><b>{int(h['body'])}% / {int(h['adx'])}</b></div>
<div class="row"><span>거래량(평균대비)</span><b>{h['vol_ratio']}배</b></div>
</div>'''
        html += '</div>'
    
    html += f'''<footer>알파 하이킨아시 = OHLC를 EMA로 평활한 뒤 표준 HA 공식을 적용한 변형입니다. ★ = 전환일 거래량이 평균을 크게 웃돈 강신호. 정보 제공용이며 투자 권유가 아닙니다. 모든 투자 책임은 본인에게 있습니다.</footer>
</div>
<script>
const fmt=n=>Math.round(n).toLocaleString('ko-KR');
function drawSpark(id,spark,flipPos){{
  const cv=document.getElementById(id); if(!cv||!spark||!spark.length)return;
  const dpr=window.devicePixelRatio||1,W=cv.clientWidth,H=60;
  cv.width=W*dpr; cv.height=H*dpr; const ctx=cv.getContext('2d'); ctx.scale(dpr,dpr);
  const cs=getComputedStyle(document.documentElement);
  const up=cs.getPropertyValue('--up').trim(),dn=cs.getPropertyValue('--down').trim(),gold=cs.getPropertyValue('--gold').trim();
  const vals=spark.map(s=>s.c),hi=Math.max(...vals),lo=Math.min(...vals),span=(hi-lo)||1;
  const n=spark.length,bw=W/n,pad=6,y=v=>pad+(H-pad*2)*(1-(v-lo)/span);
  spark.forEach((s,i)=>{{
    if(i===0)return;
    ctx.strokeStyle=s.up?up:dn; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(bw*(i-1)+bw/2,y(spark[i-1].c)); ctx.lineTo(bw*i+bw/2,y(s.c)); ctx.stroke();
  }});
  if(flipPos>=0&&flipPos<n){{
    const x=bw*flipPos+bw/2;
    ctx.strokeStyle=gold; ctx.lineWidth=1.2; ctx.setLineDash([3,2]);
    ctx.beginPath(); ctx.moveTo(x,2); ctx.lineTo(x,H-2); ctx.stroke(); ctx.setLineDash([]);
  }}
}}
const data = {hits};
data.forEach((d,idx)=>drawSpark('sp'+idx,d.spark,d.flip_pos));
</script></body></html>"""
    return html


def build_message(hits, end, html_url=None):
    date_str = end.strftime("%Y-%m-%d")
    if not hits:
        msg = (f"📊 *코스피200 상승전환 스크리너*\n"
               f"_{date_str} 기준_\n\n"
               f"오늘은 조건을 통과한 상승 전환 종목이 없습니다.")
    else:
        strong = sum(1 for h in hits if h["grade"] == "강")
        msg = (f"📊 *코스피200 상승전환 스크리너*\n"
               f"_{date_str} 기준 · {len(hits)}종목 (강★ {strong})_")
    
    if html_url:
        msg += f"\n\n🔗 [차트 보기]({html_url})"
    msg += "\n\n_참고용입니다. 투자 책임은 본인에게 있습니다._"
    return msg


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
    
    # HTML 생성
    html_content = build_html(hits, end)
    html_filename = f"result_{end.strftime('%Y%m%d')}.html"
    
    # 로컬에 저장 (GitHub Actions에서 commit할 파일)
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML 저장: {html_filename}")
    
    # GitHub 링크 생성 (raw content 주소)
    html_url = f"https://raw.githubusercontent.com/applefactorys1/kospi/main/{html_filename}"
    
    # 메시지 생성 (링크 포함)
    msg = build_message(hits, end, html_url)
    sent = send_telegram(msg)
    print("발송 완료" if sent else "발송 안 됨")


if __name__ == "__main__":
    main()
