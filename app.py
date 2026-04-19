from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage
from groq import Groq
import os
import logging
import traceback
import base64
import re
import requests
import psycopg2
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")
ADMIN_LINE_ID = os.environ.get("ADMIN_LINE_ID", "")
MONTHLY_QUOTA = 200
FREE_DAILY_QUOTA = 14

RAILWAY_API_TOKEN = os.environ.get("RAILWAY_API_TOKEN", "")
RAILWAY_PROJECT_ID = os.environ.get("RAILWAY_PROJECT_ID", "")
RAILWAY_ENVIRONMENT_ID = os.environ.get("RAILWAY_ENVIRONMENT_ID", "")
RAILWAY_SERVICE_ID = os.environ.get("RAILWAY_SERVICE_ID", "")

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_HISTORY = 10

# ── Bot instances ──────────────────────────────────────────────
math_api = LineBotApi(os.environ.get("MATH_LINE_CHANNEL_ACCESS_TOKEN", ""))
math_handler = WebhookHandler(os.environ.get("MATH_LINE_CHANNEL_SECRET", ""))
MATH_PAID = set(uid.strip() for uid in os.environ.get("MATH_PAID_USER_IDS", "").split(",") if uid.strip())

english_api = LineBotApi(os.environ.get("ENGLISH_LINE_CHANNEL_ACCESS_TOKEN", ""))
english_handler = WebhookHandler(os.environ.get("ENGLISH_LINE_CHANNEL_SECRET", ""))
ENGLISH_PAID = set(uid.strip() for uid in os.environ.get("ENGLISH_PAID_USER_IDS", "").split(",") if uid.strip())

natural_api = LineBotApi(os.environ.get("NATURAL_LINE_CHANNEL_ACCESS_TOKEN", ""))
natural_handler = WebhookHandler(os.environ.get("NATURAL_LINE_CHANNEL_SECRET", ""))
NATURAL_PAID = set(uid.strip() for uid in os.environ.get("NATURAL_PAID_USER_IDS", "").split(",") if uid.strip())

# ── System prompts ─────────────────────────────────────────────
MATH_PROMPT = """你是一個專門幫助台灣高中生解數學題的助手，針對108課綱設計。

格式規則（非常重要）：
- 只能使用純文字，不可使用 Markdown（禁止 ##、**、- 列表符號）
- 數學式用文字表示，例如：x^2、sqrt(x)、(a+b)/c
- 換行用空白行分隔段落

回答規則：
1. 用繁體中文解說
2. 解題步驟要清楚，逐步說明
3. 只回答數學相關問題

訂閱資訊（當用戶詢問訂閱、付費、升級等相關問題時告知）：
- 免費版：每天 14 則訊息，使用基礎 AI 模型
- 進階版：每月 70 元，每月 200 則訊息，使用更強的 AI 模型解題更準確，用完後降回每日 14 則免費版，下月重置
- 付款方式：第一銀行（代碼007）帳號 21257048971，或街口支付帳號 905432635
- 付款後將截圖傳至 LINE ID：a0970801250，並告知自己的 LINE ID（傳「我的ID」可查詢）
- 確認後將開通進階版"""

ENGLISH_PROMPT = """你是一個專門幫助台灣高中生學習英文的助手，針對108課綱設計。

格式規則（非常重要）：
- 只能使用純文字，不可使用 Markdown（禁止 ##、**、- 列表符號）
- 換行用空白行分隔段落

回答規則：
1. 用繁體中文解說，英文內容保留英文
2. 解說清楚，符合學測和指考格式
3. 只回答英文相關問題

功能範圍：
- 文法題：說明文法規則，指出錯誤並解釋原因
- 閱讀測驗：解析文章大意、題目考點、答題技巧
- 單字片語：解釋意思、用法、例句
- 作文批改：指出文法錯誤、用字建議、結構改善
- 翻譯：中英互譯並說明用法

訂閱資訊（當用戶詢問訂閱、付費、升級等相關問題時告知）：
- 免費版：每天 14 則訊息，使用基礎 AI 模型
- 進階版：每月 30 元，每月 200 則訊息，使用更強的 AI 模型解題更準確，用完後降回每日 14 則免費版，下月重置
- 付款方式：第一銀行（代碼007）帳號 21257048971，或街口支付帳號 905432635
- 付款後將截圖傳至 LINE ID：a0970801250，並告知自己的 LINE ID（傳「我的ID」可查詢）
- 確認後將開通進階版"""

NATURAL_PROMPT = """你是一個專門幫助台灣高中生學習自然科的助手，涵蓋物理、化學、生物、地球科學，針對108課綱設計。

格式規則（非常重要）：
- 只能使用純文字，不可使用 Markdown（禁止 ##、**、- 列表符號）
- 換行用空白行分隔段落

回答規則：
1. 用繁體中文解說，專有名詞保留原文
2. 解說清楚，符合學測和指考格式
3. 只回答自然科相關問題（物理、化學、生物、地科）

108課綱範圍：

物理：力學、電磁學、波動與光學、近代物理
化學：物質結構、化學反應、有機化學、定量分析
生物：細胞、遺傳、生態、演化與多樣性
地球科學：地球構造、大氣、海洋、天文

訂閱資訊（當用戶詢問訂閱、付費、升級等相關問題時告知）：
- 免費版：每天 14 則訊息，使用基礎 AI 模型
- 進階版：每月 50 元，每月 200 則訊息，使用更強的 AI 模型解題更準確，用完後降回每日 14 則免費版，下月重置
- 付款方式：第一銀行（代碼007）帳號 21257048971，或街口支付帳號 905432635
- 付款後將截圖傳至 LINE ID：a0970801250，並告知自己的 LINE ID（傳「我的ID」可查詢）
- 確認後將開通進階版"""

MATH_SUBSCRIBE_MSG = """訂閱進階版數學解題助手

費用：每月 70 元

付款方式：
1. 第一銀行（代碼 007）帳號 21257048971
2. 街口支付 帳號 905432635

付款後請將截圖傳送至 LINE ID：a0970801250，並告知您的 LINE ID（傳「我的ID」可查詢），確認後將為您開通進階版。

進階版功能：每月 200 則訊息，使用更強的 AI 模型解題更準確，用完後自動降回每日 14 則免費版，下月自動重置。"""

ENGLISH_SUBSCRIBE_MSG = """訂閱進階版英文解題助手

費用：每月 30 元

付款方式：
1. 第一銀行（代碼 007）帳號 21257048971
2. 街口支付 帳號 905432635

付款後請將截圖傳送至 LINE ID：a0970801250，並告知您的 LINE ID（傳「我的ID」可查詢），確認後將為您開通進階版。

進階版功能：每月 200 則訊息，使用更強的 AI 模型解題更準確，用完後自動降回每日 14 則免費版，下月自動重置。"""

NATURAL_SUBSCRIBE_MSG = """訂閱進階版自然解題助手

費用：每月 50 元

付款方式：
1. 第一銀行（代碼 007）帳號 21257048971
2. 街口支付 帳號 905432635

付款後請將截圖傳送至 LINE ID：a0970801250，並告知您的 LINE ID（傳「我的ID」可查詢），確認後將為您開通進階版。

進階版功能：每月 200 則訊息，使用更強的 AI 模型解題更準確，用完後自動降回每日 14 則免費版，下月自動重置。"""

# ── DB helpers ─────────────────────────────────────────────────
def get_db():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    user_id TEXT PRIMARY KEY,
                    count INTEGER DEFAULT 0,
                    month TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    user_id TEXT PRIMARY KEY,
                    history TEXT,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

def get_usage(user_id, period):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count, month FROM usage WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if not row or row[1] != period:
                cur.execute("""
                    INSERT INTO usage (user_id, count, month) VALUES (%s, 0, %s)
                    ON CONFLICT (user_id) DO UPDATE SET count = 0, month = %s
                """, (user_id, period, period))
                return 0
            return row[0]

def increment_usage(user_id, period):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO usage (user_id, count, month) VALUES (%s, 1, %s)
                ON CONFLICT (user_id) DO UPDATE SET count = usage.count + 1, month = %s
            """, (user_id, period, period))

def load_history(user_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT history FROM conversations WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
            return []

def save_history(user_id, history):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations (user_id, history, updated_at) VALUES (%s, %s, NOW())
                ON CONFLICT (user_id) DO UPDATE SET history = %s, updated_at = NOW()
            """, (user_id, json.dumps(history), json.dumps(history)))

try:
    init_db()
    logger.info("Database initialized")
except Exception as e:
    logger.error(f"DB init error: {e}")

# ── Railway paid user update ───────────────────────────────────
def add_paid_user(paid_set, env_var_name, new_user_id):
    paid_set.add(new_user_id)
    new_value = ",".join(paid_set)
    query = """
    mutation variableUpsert($input: VariableUpsertInput!) {
        variableUpsert(input: $input)
    }
    """
    variables = {
        "input": {
            "projectId": RAILWAY_PROJECT_ID,
            "environmentId": RAILWAY_ENVIRONMENT_ID,
            "serviceId": RAILWAY_SERVICE_ID,
            "name": env_var_name,
            "value": new_value
        }
    }
    resp = requests.post(
        "https://backboard.railway.app/graphql/v2",
        headers={"Authorization": f"Bearer {RAILWAY_API_TOKEN}", "Content-Type": "application/json"},
        json={"query": query, "variables": variables},
        timeout=10
    )
    return resp.json()

# ── AI helpers ─────────────────────────────────────────────────
def call_ai(model, messages):
    return groq_client.chat.completions.create(model=model, messages=messages)

def clean_response(text):
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_quota_period(is_paid, quota_id):
    if is_paid:
        month_period = datetime.now().strftime("%Y-%m")
        try:
            monthly_usage = get_usage(quota_id, month_period)
        except Exception:
            monthly_usage = 0
        if monthly_usage < MONTHLY_QUOTA:
            return month_period, MONTHLY_QUOTA, ""
        else:
            return datetime.now().strftime("%Y-%m-%d"), FREE_DAILY_QUOTA, f"本月 {MONTHLY_QUOTA} 則訊息已用完，已降回每日 {FREE_DAILY_QUOTA} 則免費版。"
    else:
        return datetime.now().strftime("%Y-%m-%d"), FREE_DAILY_QUOTA, f"今日免費額度（{FREE_DAILY_QUOTA} 則）已用完，明天再來或傳「訂閱」升級進階版。"

def notify_admin(line_api, msg):
    if ADMIN_LINE_ID:
        try:
            line_api.push_message(ADMIN_LINE_ID, TextSendMessage(text=f"[Bot錯誤通知]\n{msg}"))
        except Exception:
            pass

# ── Generic text/image handlers ────────────────────────────────
def handle_text(event, line_api, paid_set, quota_prefix, system_prompt, subscribe_msg, approve_env_var, image_prompt=None):
    user_id = event.source.user_id
    quota_id = f"{quota_prefix}:{user_id}"
    user_message = event.message.text
    is_paid = user_id in paid_set
    logger.info(f"[{quota_prefix}] User {user_id} ({'paid' if is_paid else 'free'}): {user_message}")

    if user_id == ADMIN_LINE_ID and user_message.strip().startswith("!approve "):
        target_id = user_message.strip().split(" ", 1)[1].strip()
        try:
            add_paid_user(paid_set, approve_env_var, target_id)
            reply = f"已開通付費版：{target_id}"
        except Exception as e:
            reply = f"開通失敗：{e}"
        try:
            line_api.reply_message(event.reply_token, TextSendMessage(text=reply))
        except Exception as ex:
            logger.error(f"Admin reply error: {ex}")
        return

    if user_message.strip() in ["我的id", "my id", "myid", "我的ID", "MY ID"]:
        try:
            line_api.reply_message(event.reply_token, TextSendMessage(text=f"你的 LINE ID 是：{user_id}"))
        except Exception as e:
            logger.error(f"ID reply error: {e}")
        return

    if user_message.strip() in ["訂閱", "subscribe", "付費", "升級"]:
        try:
            line_api.reply_message(event.reply_token, TextSendMessage(text=subscribe_msg))
        except Exception as e:
            logger.error(f"Subscribe reply error: {e}")
        return

    period, quota, quota_msg = get_quota_period(is_paid, quota_id)

    try:
        usage = get_usage(quota_id, period)
        if usage >= quota:
            line_api.reply_message(event.reply_token, TextSendMessage(text=quota_msg))
            return
    except Exception as e:
        logger.error(f"Usage check error: {e}")

    try:
        history = load_history(quota_id)
    except Exception:
        history = []

    history.append({"role": "user", "content": user_message})
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    reply_text = "抱歉，系統暫時無法回應，請稍後再試。"
    try:
        response = call_ai(TEXT_MODEL, [{"role": "system", "content": system_prompt}] + history)
        reply_text = clean_response(response.choices[0].message.content)[:4900]
        history.append({"role": "assistant", "content": reply_text})
        try:
            save_history(quota_id, history)
        except Exception as e:
            logger.error(f"Save history error: {e}")
        try:
            increment_usage(quota_id, period)
        except Exception as e:
            logger.error(f"Usage increment error: {e}")
    except Exception as e:
        logger.error(f"AI error: {e}\n{traceback.format_exc()}")
        notify_admin(line_api, f"AI error [{quota_prefix}] user {user_id}: {e}")
    try:
        line_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
    except Exception as e:
        logger.error(f"LINE reply error: {e}\n{traceback.format_exc()}")

def handle_image_msg(event, line_api, paid_set, quota_prefix, system_prompt, image_prompt):
    user_id = event.source.user_id
    quota_id = f"{quota_prefix}:{user_id}"
    is_paid = user_id in paid_set
    logger.info(f"[{quota_prefix}] User {user_id} sent image")

    period, quota, quota_msg = get_quota_period(is_paid, quota_id)

    try:
        usage = get_usage(quota_id, period)
        if usage >= quota:
            line_api.reply_message(event.reply_token, TextSendMessage(text=quota_msg))
            return
    except Exception as e:
        logger.error(f"Usage check error: {e}")

    try:
        message_content = line_api.get_message_content(event.message.id)
        image_data = b"".join(chunk for chunk in message_content.iter_content())
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        response = call_ai(VISION_MODEL, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": image_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ])
        reply_text = clean_response(response.choices[0].message.content)[:4900]
        try:
            increment_usage(quota_id, period)
        except Exception as e:
            logger.error(f"Usage increment error: {e}")
    except Exception as e:
        logger.error(f"Image error: {e}\n{traceback.format_exc()}")
        notify_admin(line_api, f"Image error [{quota_prefix}] user {user_id}: {e}")
        reply_text = "抱歉，無法處理圖片，請稍後再試。"
    try:
        line_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
    except Exception as e:
        logger.error(f"LINE reply error: {e}\n{traceback.format_exc()}")

# ── Routes ─────────────────────────────────────────────────────
@app.route("/math/callback", methods=['POST'])
def math_callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        math_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        logger.error(f"Math webhook error: {e}")
        abort(500)
    return 'OK'

@app.route("/english/callback", methods=['POST'])
def english_callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        english_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        logger.error(f"English webhook error: {e}")
        abort(500)
    return 'OK'

@app.route("/natural/callback", methods=['POST'])
def natural_callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        natural_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        logger.error(f"Natural webhook error: {e}")
        abort(500)
    return 'OK'

@app.route("/test")
def test():
    try:
        response = call_ai(TEXT_MODEL, [{"role": "user", "content": "say hi in traditional chinese"}])
        return f"OK: {response.choices[0].message.content}"
    except Exception as e:
        return f"ERROR: {e}", 500

# ── LINE event handlers ────────────────────────────────────────
@math_handler.add(MessageEvent, message=TextMessage)
def math_text(event):
    handle_text(event, math_api, MATH_PAID, "math", MATH_PROMPT, MATH_SUBSCRIBE_MSG, "MATH_PAID_USER_IDS")

@math_handler.add(MessageEvent, message=ImageMessage)
def math_image(event):
    handle_image_msg(event, math_api, MATH_PAID, "math", MATH_PROMPT, "請看這張圖片中的數學題目並解題。")

@english_handler.add(MessageEvent, message=TextMessage)
def english_text(event):
    handle_text(event, english_api, ENGLISH_PAID, "english", ENGLISH_PROMPT, ENGLISH_SUBSCRIBE_MSG, "ENGLISH_PAID_USER_IDS")

@english_handler.add(MessageEvent, message=ImageMessage)
def english_image(event):
    handle_image_msg(event, english_api, ENGLISH_PAID, "english", ENGLISH_PROMPT, "請看這張圖片中的英文題目並解析。")

@natural_handler.add(MessageEvent, message=TextMessage)
def natural_text(event):
    handle_text(event, natural_api, NATURAL_PAID, "natural", NATURAL_PROMPT, NATURAL_SUBSCRIBE_MSG, "NATURAL_PAID_USER_IDS")

@natural_handler.add(MessageEvent, message=ImageMessage)
def natural_image(event):
    handle_image_msg(event, natural_api, NATURAL_PAID, "natural", NATURAL_PROMPT, "請看這張圖片中的自然科題目並解題。")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
