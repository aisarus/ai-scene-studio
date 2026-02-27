import os
import base64
import json
import traceback
from typing import Optional, Literal, Dict, Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Optional: fallback image if no API key / provider error
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


# ---------------------------
# Config
# ---------------------------

APP_TITLE = "AI Scene Studio API"

# Provider selection:
# - If GEMINI_API_KEY is present and google-genai is installed -> use Gemini
# - Otherwise -> fallback images/text (so UI stays alive)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-2.0-flash")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.0-flash-image-generation")  # override in Render env if needed

# Chroma key default (green)
DEFAULT_KEY_COLOR = os.getenv("KEY_COLOR", "#00FF00")

STATIC_DIR = os.getenv("STATIC_DIR", "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")


# ---------------------------
# Gemini client (lazy)
# ---------------------------

_gemini_client = None
_gemini_import_error = None


def get_gemini_client():
    global _gemini_client, _gemini_import_error
    if _gemini_client is not None:
        return _gemini_client

    if not GEMINI_API_KEY:
        _gemini_import_error = "GEMINI_API_KEY / GOOGLE_API_KEY is not set"
        return None

    try:
        # New SDK: google-genai
        # pip: google-genai
        from google import genai  # type: ignore
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        return _gemini_client
    except Exception as e:
        _gemini_import_error = f"Failed to import/init google-genai: {type(e).__name__}: {e}"
        return None


# ---------------------------
# Models
# ---------------------------

class PingResponse(BaseModel):
    ok: bool
    provider: str
    text_model: str
    image_model: str
    static_dir: str


class ImprovePromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    kind: Literal["scene", "background", "subject"] = "scene"


class ImprovePromptResponse(BaseModel):
    improved: str
    notes: str
    used_provider: str


class GenerateTextRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class GenerateTextResponse(BaseModel):
    text: str
    used_provider: str


class GenerateLayerRequest(BaseModel):
    layer_kind: Literal["background", "subject"] = "background"
    prompt: str = Field(..., min_length=1)
    key_color: Optional[str] = None  # e.g. "#00FF00"
    # optional controls passed by UI; server doesn't need them but keeps for future
    style: Optional[str] = None
    aspect: Optional[str] = None  # "1:1", "16:9", etc.


class GenerateLayerResponse(BaseModel):
    image_base64: str
    mime_type: str = "image/png"
    layer_kind: str
    key_color: str
    final_prompt: str
    used_provider: str
    used_fallback: bool
    debug: Optional[Dict[str, Any]] = None


# ---------------------------
# Helpers
# ---------------------------

def _safe_json_error(message: str, status: int = 400, debug: Optional[dict] = None):
    payload = {"error": message}
    if debug:
        payload["debug"] = debug
    return JSONResponse(payload, status_code=status)


def _make_fallback_png(text: str) -> str:
    """
    Generates a simple PNG with text and returns base64 (no external assets).
    """
    W, H = 1024, 640
    img = Image.new("RGB", (W, H), (18, 22, 28))
    d = ImageDraw.Draw(img)

    # Try to use a default font; if not available, pillow will use its own
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad = 28
    lines = []
    s = text.strip()
    if not s:
        s = "No text"
    # naive wrap
    max_chars = 70
    while len(s) > max_chars:
        lines.append(s[:max_chars])
        s = s[max_chars:]
    lines.append(s)

    y = pad
    d.text((pad, y), "Fallback image (no provider / error)", fill=(180, 200, 220), font=font)
    y += 28

    for line in lines[:18]:
        d.text((pad, y), line, fill=(235, 235, 235), font=font)
        y += 24

    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def improve_prompt_local(prompt: str, kind: str) -> ImprovePromptResponse:
    """
    Honest local "improver": not magic, just adds structure and constraints.
    """
    p = prompt.strip()

    if kind == "background":
        improved = (
            f"{p}\n\n"
            "Constraints:\n"
            "- Background only: NO characters, NO objects in foreground.\n"
            "- High quality, coherent lighting.\n"
            "- No text, no watermark, no logos.\n"
        )
        notes = "Local rules: background-only constraints + quality/no-text."
    elif kind == "subject":
        improved = (
            f"{p}\n\n"
            "Constraints:\n"
            "- Single subject only, centered, fully visible (not cropped).\n"
            f"- Solid flat chroma background color ONLY ({DEFAULT_KEY_COLOR}), no gradient, no shadows.\n"
            "- No extra objects, no scenery, no text, no watermark.\n"
            "- Clean edges.\n"
        )
        notes = "Local rules: single-subject constraints + chroma background + no extras."
    else:
        improved = (
            f"{p}\n\n"
            "Output requirements:\n"
            "- One clear scene.\n"
            "- Specify style, lighting, camera angle.\n"
            "- No text, no watermark.\n"
        )
        notes = "Local rules: structured prompt (style/lighting/camera) + no-text."
    return ImprovePromptResponse(improved=improved, notes=notes, used_provider="local")


def build_layer_prompt(req: GenerateLayerRequest) -> str:
    base = req.prompt.strip()
    key_color = (req.key_color or DEFAULT_KEY_COLOR).upper()

    if req.layer_kind == "background":
        return (
            f"{base}\n\n"
            "Rules:\n"
            "- Background plate only.\n"
            "- No characters, no main subject in foreground.\n"
            "- No text, no watermark.\n"
            "- Photorealistic or coherent illustration (follow user's style), consistent lighting.\n"
        )

    # subject layer
    return (
        f"{base}\n\n"
        "STRICT RULES:\n"
        "- ONE main subject only.\n"
        "- Centered, fully visible (not cropped).\n"
        f"- Background must be a perfectly solid flat color: {key_color} (no gradient, no shadow, no texture).\n"
        "- No additional objects, no scenery.\n"
        "- No text, no watermark.\n"
        "- Clean edges.\n"
    )


def gemini_generate_text(prompt: str) -> str:
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(_gemini_import_error or "Gemini client not available")

    # google-genai API style (new)
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
    )

    # Try common fields
    txt = getattr(resp, "text", None)
    if txt:
        return txt

    # fallback: parse candidates
    cands = getattr(resp, "candidates", None) or []
    parts_out = []
    for c in cands:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) or []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                parts_out.append(t)
    return "\n".join(parts_out).strip()


def gemini_generate_image_base64(prompt: str) -> Dict[str, str]:
    """
    Returns dict with keys: image_base64, mime_type.
    Works only if provider returns inline_data.
    """
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(_gemini_import_error or "Gemini client not available")

    resp = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=[prompt],
    )

    # Search inline_data in response candidates/parts
    cands = getattr(resp, "candidates", None) or []
    for c in cands:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data
                if isinstance(data, (bytes, bytearray)):
                    data = base64.b64encode(data).decode("utf-8")
                mime = getattr(inline, "mime_type", "image/png")
                return {"image_base64": data, "mime_type": mime}

    # Some SDK versions expose parts directly
    parts = getattr(resp, "parts", None) or []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            if isinstance(data, (bytes, bytearray)):
                data = base64.b64encode(data).decode("utf-8")
            mime = getattr(inline, "mime_type", "image/png")
            return {"image_base64": data, "mime_type": mime}

    raise RuntimeError("Image not generated (no inline_data in response).")


# ---------------------------
# App
# ---------------------------

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/ping", response_model=PingResponse)
def ping():
    provider = "gemini" if (get_gemini_client() is not None) else "local"
    return PingResponse(
        ok=True,
        provider=provider,
        text_model=TEXT_MODEL,
        image_model=IMAGE_MODEL,
        static_dir=STATIC_DIR,
    )


@app.post("/api/improve_prompt", response_model=ImprovePromptResponse)
def improve_prompt(req: ImprovePromptRequest):
    # Try provider first, fallback to local rules
    client = get_gemini_client()
    if client is None:
        return improve_prompt_local(req.prompt, req.kind)

    # Provider-based improver: still "честно", просто просим модель переформулировать
    # и добавляем ограничения по виду слоя.
    local = improve_prompt_local(req.prompt, req.kind)
    meta = (
        "You are a prompt improver. Keep the user's meaning. "
        "Make it more specific and production-ready. "
        "Return ONLY the improved prompt, no commentary."
    )
    prompt = f"{meta}\n\nUSER PROMPT:\n{req.prompt}\n\nHARD CONSTRAINTS TO INCLUDE:\n{local.improved.split('Constraints:',1)[-1] if 'Constraints:' in local.improved else ''}"
    try:
        improved = gemini_generate_text(prompt).strip()
        if not improved:
            return local
        return ImprovePromptResponse(improved=improved, notes="Gemini rewrite + constraints.", used_provider="gemini")
    except Exception:
        return local


@app.post("/api/generate_text", response_model=GenerateTextResponse)
def generate_text(req: GenerateTextRequest):
    client = get_gemini_client()
    if client is None:
        # Local fallback: just echo (so UI doesn't die)
        return GenerateTextResponse(text=req.prompt.strip(), used_provider="local")

    try:
        text = gemini_generate_text(req.prompt).strip()
        return GenerateTextResponse(text=text or "", used_provider="gemini")
    except Exception as e:
        return GenerateTextResponse(text=f"[error] {type(e).__name__}: {e}", used_provider="gemini")


@app.post("/api/generate_layer", response_model=GenerateLayerResponse)
def generate_layer(req: GenerateLayerRequest):
    final_prompt = build_layer_prompt(req)
    key_color = (req.key_color or DEFAULT_KEY_COLOR).upper()

    # Try Gemini image generation, fallback to generated PNG
    client = get_gemini_client()
    if client is None:
        img_b64 = _make_fallback_png(f"{req.layer_kind.upper()} layer\n\n{final_prompt}")
        return GenerateLayerResponse(
            image_base64=img_b64,
            mime_type="image/png",
            layer_kind=req.layer_kind,
            key_color=key_color,
            final_prompt=final_prompt,
            used_provider="local",
            used_fallback=True,
            debug={"reason": _gemini_import_error},
        )

    try:
        out = gemini_generate_image_base64(final_prompt)
        return GenerateLayerResponse(
            image_base64=out["image_base64"],
            mime_type=out.get("mime_type", "image/png"),
            layer_kind=req.layer_kind,
            key_color=key_color,
            final_prompt=final_prompt,
            used_provider="gemini",
            used_fallback=False,
            debug={"model": IMAGE_MODEL},
        )
    except Exception as e:
        # Fallback image so UI keeps working
        img_b64 = _make_fallback_png(f"ERROR -> fallback\n{type(e).__name__}: {e}\n\n{final_prompt}")
        return GenerateLayerResponse(
            image_base64=img_b64,
            mime_type="image/png",
            layer_kind=req.layer_kind,
            key_color=key_color,
            final_prompt=final_prompt,
            used_provider="gemini",
            used_fallback=True,
            debug={
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=2),
                "model": IMAGE_MODEL,
            },
        )


# ---------------------------
# Static + index
# ---------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    # Serve index.html explicitly (more reliable than relying on html=True mount alone)
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Missing static/index.html</h1>", status_code=500)


# Mount static directory at /static
# (This avoids the classic 'app.mount(...)def ...' paste disaster and keeps routes clean.)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")    allow_credentials=True,
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
