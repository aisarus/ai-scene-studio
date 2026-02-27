import os
import base64
from typing import Optional, Literal, Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Gemini (google-genai)
try:
    from google import genai
except Exception:
    genai = None


# -------------------------
# Config
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-2.0-flash")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.0-flash-image-generation")
DEFAULT_KEY_COLOR = os.getenv("DEFAULT_KEY_COLOR", "#00FF00")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")


# -------------------------
# App
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo mode
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_client = None
if genai is not None and GEMINI_API_KEY:
    try:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        _client = None


# -------------------------
# Request models
# -------------------------
class TextRequest(BaseModel):
    prompt: str


class ImageRequest(BaseModel):
    prompt: str


class LayerRequest(BaseModel):
    prompt: str
    layer_kind: Literal["background", "object"] = "background"
    key_color: Optional[str] = None
    layer_name: Optional[str] = None


# -------------------------
# Helpers
# -------------------------
def jerror(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


def require_client():
    if _client is None:
        if genai is None:
            return jerror("google-genai library is not available. Check requirements.txt.", 500)
        if not GEMINI_API_KEY:
            return jerror("GEMINI_API_KEY is not set in Render Environment Variables.", 500)
        return jerror("Gemini client init failed. Check logs.", 500)
    return None


def extract_text(resp: Any) -> str:
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    parts: List[str] = []
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        for p in (getattr(content, "parts", None) or []):
            pt = getattr(p, "text", None)
            if isinstance(pt, str) and pt.strip():
                parts.append(pt.strip())

    return "\n".join(parts).strip()


def extract_inline_image(resp: Any) -> Optional[Dict[str, str]]:
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue

            data = getattr(inline, "data", None)
            if not data:
                continue

            if isinstance(data, (bytes, bytearray)):
                b64 = base64.b64encode(data).decode("utf-8")
            elif isinstance(data, str):
                b64 = data
            else:
                continue

            mime = getattr(inline, "mime_type", None) or "image/png"
            return {"image_base64": b64, "mime_type": mime}

    return None


def normalize_hex_color(s: str) -> str:
    s2 = (s or "").strip()
    if not s2:
        return DEFAULT_KEY_COLOR
    if not s2.startswith("#"):
        s2 = "#" + s2
    if len(s2) not in (4, 7):
        return DEFAULT_KEY_COLOR
    return s2.upper()


# -------------------------
# Routes
# -------------------------
@app.get("/")
def index():
    if not os.path.exists(INDEX_PATH):
        return jerror("static/index.html not found. Create it in the repo.", 500)
    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get("/health")
def health():
    ok = True
    reason = None

    if genai is None:
        ok = False
        reason = "google-genai not installed"
    elif not GEMINI_API_KEY:
        ok = False
        reason = "GEMINI_API_KEY missing"
    elif _client is None:
        ok = False
        reason = "client init failed"

    return {"ok": ok, "reason": reason}


@app.post("/api/generate_text")
def generate_text(req: TextRequest):
    c = require_client()
    if c:
        return c

    prompt = (req.prompt or "").strip()
    if not prompt:
        return jerror("Empty prompt.")

    try:
        resp = _client.models.generate_content(
            model=TEXT_MODEL,
            contents=[prompt],
        )
        text = extract_text(resp)
        if not text:
            return jerror("No text returned by model.", 502)
        return {"text": text}
    except Exception as e:
        return jerror(f"Text generation failed: {type(e).__name__}: {e}", 500)


@app.post("/api/generate_image")
def generate_image(req: ImageRequest):
    c = require_client()
    if c:
        return c

    prompt = (req.prompt or "").strip()
    if not prompt:
        return jerror("Empty prompt.")

    try:
        resp = _client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[prompt],
        )
        img = extract_inline_image(resp)
        if not img:
            return jerror("Image not generated (no inline_data).", 502)
        return img
    except Exception as e:
        return jerror(f"Image generation failed: {type(e).__name__}: {e}", 500)


@app.post("/api/generate_layer")
def generate_layer(req: LayerRequest):
    c = require_client()
    if c:
        return c

    base = (req.prompt or "").strip()
    if not base:
        return jerror("Empty prompt.")

    kind = req.layer_kind
    key = normalize_hex_color(req.key_color or DEFAULT_KEY_COLOR)

    if kind == "background":
        final_prompt = (
            f"{base}\n\n"
            "Rules:\n"
            "- Background-only plate. No main subject in foreground.\n"
            "- No text, no watermark.\n"
            "- Clean, coherent lighting.\n"
        )
    else:
        final_prompt = (
            f"{base}\n\n"
            "STRICT RULES:\n"
            "- Exactly ONE object/character.\n"
            "- Centered and fully visible (not cropped).\n"
            f"- Background is a perfectly solid flat color {key} (no gradient, no shadow, no texture).\n"
            "- No extra objects, no scenery.\n"
            "- No text, no watermark.\n"
            "- Clean edges.\n"
        )

    try:
        resp = _client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[final_prompt],
        )
        img = extract_inline_image(resp)
        if not img:
            return jerror("Layer image not generated (no inline_data).", 502)

        return {
            **img,
            "layer_kind": kind,
            "layer_name": req.layer_name,
            "key_color": key if kind == "object" else None,
        }
    except Exception as e:
        return jerror(f"Layer generation failed: {type(e).__name__}: {e}", 500)

client = None
if genai and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None


# ----------------------------
# Models
# ----------------------------

class TextRequest(BaseModel):
    prompt: str


class ImageRequest(BaseModel):
    prompt: str


class LayerRequest(BaseModel):
    prompt: str
    layer_kind: Literal["background", "object"] = "background"
    layer_name: Optional[str] = None
    key_color: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------

def error(msg: str) -> Dict[str, str]:
    return {"error": msg}


def extract_image(resp: Any):
    candidates = getattr(resp, "candidates", []) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", []) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data
                if isinstance(data, (bytes, bytearray)):
                    data = base64.b64encode(data).decode("utf-8")
                return {
                    "image_base64": data,
                    "mime_type": getattr(inline, "mime_type", "image/png"),
                }
    return None


def ensure_client():
    if client is None:
        return error("GEMINI_API_KEY missing or Gemini client failed.")
    return None


# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate_text")
def generate_text(req: TextRequest):
    c = ensure_client()
    if c:
        return c

    prompt = req.prompt.strip()
    if not prompt:
        return error("Empty prompt.")

    try:
        resp = client.models.generate_content(
            model=TEXT_MODEL,
            contents=[prompt],
        )

        text = getattr(resp, "text", None)
        if text:
            return {"text": text}

        parts = []
        for cand in getattr(resp, "candidates", []) or []:
            for p in getattr(cand.content, "parts", []) or []:
                if getattr(p, "text", None):
                    parts.append(p.text)

        return {"text": "\n".join(parts)}

    except Exception as e:
        return error(f"Text generation failed: {type(e).__name__}: {e}")


@app.post("/generate_image")
def generate_image(req: ImageRequest):
    c = ensure_client()
    if c:
        return c

    prompt = req.prompt.strip()
    if not prompt:
        return error("Empty prompt.")

    try:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[prompt],
        )

        img = extract_image(resp)
        if img:
            return img

        return error("Image not generated.")

    except Exception as e:
        return error(f"Image generation failed: {type(e).__name__}: {e}")


@app.post("/generate_layer")
def generate_layer(req: LayerRequest):
    c = ensure_client()
    if c:
        return c

    base = req.prompt.strip()
    if not base:
        return error("Empty prompt.")

    key_color = req.key_color or KEY_COLOR_DEFAULT

    if req.layer_kind == "background":
        final_prompt = (
            f"{base}\n\n"
            "- Background only\n"
            "- No main subject in foreground\n"
            "- No text, no watermark"
        )
    else:
        final_prompt = (
            f"{base}\n\n"
            "- Exactly ONE object\n"
            "- Centered and fully visible\n"
            f"- Solid flat background color {key_color}\n"
            "- No extra objects\n"
            "- No text, no watermark"
        )

    try:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[final_prompt],
        )

        img = extract_image(resp)
        if img:
            return {
                **img,
                "layer_kind": req.layer_kind,
                "layer_name": req.layer_name,
                "key_color": key_color,
            }

        return error("Layer not generated.")

    except Exception as e:
        return error(f"Layer generation failed: {type(e).__name__}: {e}")
