import os
import requests
from datetime import datetime

# =========================
# 텔레그램 설정
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT")


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("텔레그램 토큰 또는 CHAT ID가 없습니다.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    data = {
        "chat_id": TELEGRAM_CHAT,
        "text": message,
        "parse_mode": "HTML"
    }

    response = requests.post(url, data=data)
    print("텔레그램 응답:", response.status_code, response.text)


# =========================
# 여기부터 네 기존 스캔 로직
# =========================
def scan_market():
    """
    조건 충족 종목을 찾는 함수.
    기존에 네가 쓰던 종목 검색 로직이 있으면
    이 함수 안에 넣으면 됨.
    """

    picks = []

    # 예시 구조
    # 조건 맞는 종목이 있으면 아래처럼 추가
    #
    # picks.append({
    #     "name": "HL만도",
    #     "code": "204320",
    #     "price": 74600,
    #     "reason": "알파 하이킨아시 상승전환"
    # })

    return picks


# =========================
# 메시지 만들기
# =========================
def make_message(picks):
    today = datetime.now().strftime("%Y-%m-%d")

    message = f"📈 <b>코스피200 상승전환 알림</b>\n"
    message += f"날짜: {today}\n\n"
    message += f"조건 충족 종목: {len(picks)}개\n\n"

    for i, item in enumerate(picks, start=1):
        name = item.get("name", "")
        code = item.get("code", "")
        price = item.get("price", "")
        reason = item.get("reason", "")

        message += f"{i}. <b>{name}</b> ({code})\n"
        if price:
            message += f"현재가: {price:,}원\n"
        if reason:
            message += f"신호: {reason}\n"
        message += "\n"

    return message


# =========================
# 실행부
# =========================
if __name__ == "__main__":
    print("===== 코스피200 상승전환 스캔 시작 =====")

    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    picks = scan_market()

    if not picks:
        message = (
            "📭 <b>코스피200 상승전환 알림</b>\n"
            f"시간: {today}\n\n"
            "오늘 조건 충족 종목 없음"
        )
        send_telegram(message)
        print("조건 충족 종목 없음")
    else:
        message = make_message(picks)
        send_telegram(message)
        print(f"조건 충족 종목 {len(picks)}개 발송 완료")

    print("===== 스캔 종료 =====")
