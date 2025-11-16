from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
import datetime
import os
import json
import httpx
from typing import Dict, Optional, List
from google.oauth2 import service_account
from googleapiclient.discovery import build

app = FastAPI()

# ============================================================
# SHOP CONFIG
# ============================================================

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


# ============================================================
# GOOGLE CALENDAR
# ============================================================

def get_calendar_service():
    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_path or not os.path.exists(sa_path):
        return None
    creds = service_account.Credentials.from_service_account_file(
        sa_path, scopes=["https://www.googleapis.com/auth/calendar"]
    )
    return build("calendar", "v3", credentials=creds)


def create_calendar_event(shop: ShopConfig, start_dt, end_dt, phone: str):
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
        calendarId=shop.calendar_id,
        body=event
    ).execute()

    return created.get("id")


# ============================================================
# MULTI-IMAGE EXTRACTION
# ============================================================

def extract_image_urls(form) -> List[str]:
    urls = []
    i = 0
    while True:
        key = f"MediaUrl{i}"
        url = form.get(key)
        if not url:
            break
        urls.append(url)
        i += 1
    return urls


# ============================================================
# AI DAMAGE ESTIMATION (MULTI-IMAGE, REAL MODEL)
# ============================================================

async def estimate_damage_from_images(image_urls: list, shop: ShopConfig):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "severity": "Moderate",
            "min_cost": 450,
            "max_cost": 1200,
            "damage_areas": [],
            "damage_types": [],
            "suggested_repairs": [],
            "confidence": 0.65
        }

    prompt = """
You are an auto body damage estimator with 15+ years of real experience.
Multiple photos of the same vehicle are provided.

Analyze ALL photos and produce ONE unified JSON result using this schema:

{
  "severity": "Minor | Moderate | Severe",
  "damage_areas": ["front bumper", "rear door", "fender", "hood"],
  "damage_types": ["dent", "scratch", "crack", "paint chip", "rust", "misalignment"],
  "suggested_repairs": ["PDR", "panel replacement", "paint respray"],
  "min_cost": number,
  "max_cost": number,
  "confidence": number  (0.0 - 1.0)
}

Rules:
- Merge damage from ALL images.
- Include any damage visible in ANY image.
- Always return valid JSON.
"""

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Build content blocks for multiple images
    content = [{"type": "text", "text": "Analyze all uploaded images."}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        async with httpx.AsyncClient(timeout=35) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

        result = json.loads(resp.json()["choices"][0]["message"]["content"])

        # Safe defaults
        result.setdefault("severity", "Moderate")
        result.setdefault("min_cost", 450)
        result.setdefault("max_cost", 1200)
        result.setdefault("damage_areas", [])
        result.setdefault("damage_types", [])
        result.setdefault("suggested_repairs", [])
        result.setdefault("confidence", 0.70)

        return result

    except Exception as e:
        print("AI Estimator Error:", e)
        return {
            "severity": "Moderate",
            "min_cost": 450,
            "max_cost": 1200,
            "damage_areas": [],
            "damage_types": [],
            "suggested_repairs": [],
            "confidence": 0.55
        }


# ============================================================
# APPOINTMENT SLOTS
# ============================================================

def get_appointment_slots(n: int = 3):
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    hours = [9, 11, 14, 16]

    slots = []
    for h in hours:
        dt = tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        if dt > now:
            slots.append(dt)
    return slots[:n]


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "Backend is running!"}


@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig = Depends(get_shop)):
    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()

    image_urls = extract_image_urls(form)

    reply = MessagingResponse()

    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # --------------------------------------------------------
    # BOOKING SELECTION (1,2,3)
    # --------------------------------------------------------
    if session and session.get("awaiting_time") and body in {"1", "2", "3"}:
        idx = int(body) - 1
        slots = session["slots"]

        if 0 <= idx < len(slots):
            chosen = slots[idx]

            # Optional Google Calendar
            create_calendar_event(
                shop=shop,
                start_dt=chosen,
                end_dt=chosen + datetime.timedelta(minutes=45),
                phone=from_number
            )

            reply.message(
                f"Your appointment is booked for {chosen.strftime('%a %b %d at %I:%M %p')}."
            )

            session["awaiting_time"] = False
            SESSIONS[session_key] = session

            return Response(content=str(reply), media_type="application/xml")

    # --------------------------------------------------------
    # MULTI-IMAGE AI ESTIMATE
    # --------------------------------------------------------
    if image_urls:
        result = await estimate_damage_from_images(image_urls, shop)

        severity = result["severity"]
        min_cost = result["min_cost"]
        max_cost = result["max_cost"]
        cost_range = f"${min_cost:,.0f} – ${max_cost:,.0f}"

        areas = ", ".join(result["damage_areas"]) if result["damage_areas"] else "General Damage"
        types = ", ".join(result["damage_types"]) if result["damage_types"] else "Unspecified"

        slots = get_appointment_slots()
        SESSIONS[session_key] = {"awaiting_time": True, "slots": slots}

        lines = [
            f"🔎 AI Damage Estimate for {shop.name}",
            f"Severity: {severity}",
            f"Damaged Areas: {areas}",
            f"Damage Types: {types}",
            f"Estimated Cost: {cost_range}",
            f"Confidence: {result['confidence']:.2f}",
            "",
            "Reply with a number to book an in-person estimate:"
        ]

        for i, s in enumerate(slots, 1):
            lines.append(f"{i}) {s.strftime('%a %b %d at %I:%M %p')}")

        reply.message("\n".join(lines))
        return Response(content=str(reply), media_type="application/xml")

    # --------------------------------------------------------
    # DEFAULT PROMPT
    # --------------------------------------------------------
    intro = [
        f"Thanks for messaging {shop.name}! 👋",
        "",
        "Send 1–5 photos of the damage for an instant AI estimate."
    ]

    reply.message("\n".join(intro))
    return Response(content=str(reply), media_type="application/xml")
