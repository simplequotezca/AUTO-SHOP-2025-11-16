from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
import datetime
import os
import json
import httpx
from typing import Dict, Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build

app = FastAPI()

# ------------------------------------------------
# SHOP CONFIG
# ------------------------------------------------

class ShopConfig(BaseModel):
    id: str
    name: str
    calendar_id: Optional[str] = None
    webhook_token: str

def load_shops() -> Dict[str, ShopConfig]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        return {}
    data = json.loads(raw)
    return {s["webhook_token"]: ShopConfig(**s) for s in data}

SHOPS_BY_TOKEN: Dict[str, ShopConfig] = load_shops()
SESSIONS: Dict[str, dict] = {}

def get_shop(request: Request) -> ShopConfig:
    if not SHOPS_BY_TOKEN:
        return ShopConfig(id="default", name="Auto Body Shop", calendar_id=None, webhook_token="")
    token = request.query_params.get("token")
    if not token or token not in SHOPS_BY_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing shop token")
    return SHOPS_BY_TOKEN[token]


# ------------------------------------------------
# GOOGLE CALENDAR INTEGRATION
# ------------------------------------------------

def get_calendar_service():
    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_path or not os.path.exists(sa_path):
        return None
    creds = service_account.Credentials.from_service_account_file(
        sa_path, scopes=["https://www.googleapis.com/auth/calendar"]
    )
    return build("calendar", "v3", credentials=creds)

def create_calendar_event(shop: ShopConfig, start_dt, end_dt, phone):
    service = get_calendar_service()
    if not service or not shop.calendar_id:
        return None

    event = {
        "summary": f"Estimate appointment - {shop.name}",
        "description": f"Customer phone: {phone}",
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/Toronto"},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": "America/Toronto"},
    }

    created = service.events().insert(
        calendarId=shop.calendar_id, body=event
    ).execute()

    return created.get("id")


# ------------------------------------------------
# AI DAMAGE ESTIMATE
# ------------------------------------------------

async def estimate_damage_from_image(media_url: str, shop: ShopConfig):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Moderate", "$450–$1,200"

    prompt = (
        "You are an estimator for an auto body shop. "
        "Classify damage severity (Minor/Moderate/Severe) and estimate cost in CAD. "
        "Return JSON with severity, min_cost, max_cost."
    )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", 
             "content": [{"type": "image_url", "image_url": {"url": media_url}}]
            }
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
        severity = parsed.get("severity", "Moderate")
        min_cost = parsed.get("min_cost")
        max_cost = parsed.get("max_cost")

        if isinstance(min_cost, (int, float)) and isinstance(max_cost, (int, float)):
            cost_range = f"${min_cost:,.0f}–${max_cost:,.0f}"
        else:
            cost_range = "$450–$1,200"

        return severity, cost_range

    except:
        return "Moderate", "$450–$1,200"


# ------------------------------------------------
# APPOINTMENT SLOTS
# ------------------------------------------------

def get_appointment_slots(n: int = 3):
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    hours = [9, 11, 14, 16]
    slots = []

    for h in hours:
        slot = tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        if slot > now:
            slots.append(slot)
    return slots[:n]


# ------------------------------------------------
# ROUTES
# ------------------------------------------------

@app.get("/")
def root():
    return {"status": "Backend is running!"}

@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig = Depends(get_shop)):
    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()
    media_url = form.get("MediaUrl0")

    reply = MessagingResponse()
    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # ----------------------------
    # BOOKING SELECTION (1,2,3)
    # ----------------------------
    if session and session.get("awaiting_time") and body in {"1", "2", "3"}:
        idx = int(body) - 1
        slots = session["slots"]

        if 0 <= idx < len(slots):
            chosen = slots[idx]

            create_calendar_event(
                shop=shop,
                start_dt=chosen,
                end_dt=chosen + datetime.timedelta(minutes=45),
                phone=from_number,
            )

            reply.message(
                f"Great, {shop.name} has booked you for {chosen.strftime('%a %b %d at %I:%M %p')}."
            )

            session["awaiting_time"] = False
            SESSIONS[session_key] = session

            return Response(content=str(reply), media_type="application/xml")

    # ----------------------------
    # AI IMAGE ESTIMATE
    # ----------------------------
    if media_url:
        severity, cost_range = await estimate_damage_from_image(media_url, shop)
        slots = get_appointment_slots()

        SESSIONS[session_key] = {"awaiting_time": True, "slots": slots}

        lines = [
            f"{shop.name} - AI Damage Estimate",
            f"Severity: {severity}",
            f"Estimated Repair Range: {cost_range}",
            "",
            "Reply with a number to book an in-person estimate:",
        ]

        for i, s in enumerate(slots, 1):
            lines.append(f"{i}) {s.strftime('%a %b %d at %I:%M %p')}")

        reply.message("\n".join(lines))
        return Response(content=str(reply), media_type="application/xml")

    # ----------------------------
    # DEFAULT RESPONSE
    # ----------------------------
    intro = [
        f"Thanks for messaging {shop.name}! 👋",
        "",
        "Send a clear photo of the damage to receive an AI estimate.",
    ]

    reply.message("\n".join(intro))
    return Response(content=str(reply), media_type="application/xml")
