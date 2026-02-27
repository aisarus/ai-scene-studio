import os
import base64
from typing import Optional, Literal, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from google import genai
except Exception:
    genai = None


# ----------------------------
# Config
# ----------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEXT_MODEL = "gemini-2.0-flash"
IMAGE_MODEL = "gemini-2.0-flash-image-generation"
KEY_COLOR_DEFAULT = "#00FF00"

# ----------------------------
# App
# ----------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        return error(f"Layer generation failed: {type(e).__name__}: {e}")    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None
if genai is not None and GEMINI_API_KEY:
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
    key_color: Optional[str] = None  # used only for object layer


# ----------------------------
# Helpers
# ----------------------------
def _err(msg: str) -> Dict[str, str]:
    return {"error": msg}


def _extract_inline_image(resp: Any) -> Optional[Dict[str, str]]:
    """
    Tries to extract inline image bytes from Gemini response and return:
    {"image_base64": "...", "mime_type": "image/png"}
    Supports multiple possible response shapes.
    """
    # 1) Preferred path: resp.candidates[].content.parts[].inline_data
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data
                if isinstance(data, (bytes, bytearray)):
                    data = base64.b64encode(data).decode("utf-8")
                mime = getattr(inline, "mime_type", "image/png")
                return {"image_base64": str(data), "mime_type": str(mime)}

            # Some variants store it directly
            inline2 = getattr(part, "inlineData", None)
            if inline2 and getattr(inline2, "data", None):
                data = inline2.data
                if isinstance(data, (bytes, bytearray)):
                    data = base64.b64encode(data).decode("utf-8")
                mime = getattr(inline2, "mime_type", "image/png")
                return {"image_base64": str(data), "mime_type": str(mime)}

    # 2) Alternative: resp.parts[] (rare)
    parts2 = getattr(resp, "parts", None) or []
    for part in parts2:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            if isinstance(data, (bytes, bytearray)):
                data = base64.b64encode(data).decode("utf-8")
            mime = getattr(inline, "mime_type", "image/png")
            return {"image_base64": str(data), "mime_type": str(mime)}

    return None


def _ensure_client():
    if client is None:
        return _err("GEMINI_API_KEY is not set (or Gemini client init failed). Set it in Render Environment Variables.")
    return None


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate_text")
def generate_text(req: TextRequest):
    c_err = _ensure_client()
    if c_err:
        return c_err

    prompt = (req.prompt or "").strip()
    if not prompt:
        return _err("Empty prompt.")

    try:
        resp = client.models.generate_content(
            model=TEXT_MODEL,
            contents=[prompt],
        )

        # Common: resp.text
        text = getattr(resp, "text", None)
        if text:
            return {"text": str(text)}

        # Fallback: candidates/parts
        out_parts = []
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    out_parts.append(str(t))

        return {"text": "\n".join(out_parts).strip()}

    except Exception as e:
        return _err(f"Text generation failed: {type(e).__name__}: {e}")


@app.post("/generate_image")
def generate_image(req: ImageRequest):
    c_err = _ensure_client()
    if c_err:
        return c_err

    prompt = (req.prompt or "").strip()
    if not prompt:
        return _err("Empty prompt.")

    try:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[prompt],
        )

        img = _extract_inline_image(resp)
        if img:
            return img

        return _err("Image not generated (no inline_data in response). Try simplifying the prompt.")

    except Exception as e:
        return _err(f"Image generation failed: {type(e).__name__}: {e}")


@app.post("/generate_layer")
def generate_layer(req: LayerRequest):
    """
    Pseudo-layer generation:
      - background: prompt + constraints (no subject in foreground)
      - object: prompt + strict rules (single subject, solid chroma key background)
    """
    c_err = _ensure_client()
    if c_err:
        return c_err

    base_prompt = (req.prompt or "").strip()
    if not base_prompt:
        return _err("Empty prompt.")

    layer_kind = (req.layer_kind or "background").lower().strip()
    key_color = (req.key_color or KEY_COLOR_DEFAULT).strip()

    if layer_kind == "background":
        final_prompt = (
            f"{base_prompt}\n\n"
            "RULES:\n"
            "- Background-only image.\n"
            "- No main subject/character in foreground.\n"
            "- High quality, coherent lighting.\n"
            "- No text, no watermark.\n"
        )
    else:
        # object
        final_prompt = (
            f"{base_prompt}\n\n"
            "STRICT RULES:\n"
            "- Exactly ONE main object/character only.\n"
            "- Place it centered, fully visible (not cropped).\n"
            f"- Background must be a perfectly solid flat color {key_color} (no gradient, no shadows on background).\n"
            "- No additional objects, no scenery.\n"
            "- No text, no watermark.\n"
            "- Keep edges clean.\n"
        )

    try:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[final_prompt],
        )

        img = _extract_inline_image(resp)
        if img:
            # enrich response with layer metadata
            return {
                "image_base64": img["image_base64"],
                "mime_type": img.get("mime_type", "image/png"),
                "layer_kind": layer_kind,
                "layer_name": req.layer_name,
                "key_color": key_color,
                "used_prompt": final_prompt,  # helps debug what was sent
            }

        return _err("Layer image not generated (no inline_data in response). Try simplifying the prompt.")

    except Exception as e:
        return _err(f"Layer generation failed: {type(e).__name__}: {e}")


# ----------------------------
# Static hosting (KEEP LAST)
# ----------------------------
# This MUST be the last executable line(s) so it never glues to a def.
app.mount("/", StaticFiles(directory="static", html=True), name="static")class TextRequest(BaseModel):
    prompt: str


class ImageRequest(BaseModel):
    prompt: str


# =========================
# TEXT GENERATION
# =========================

@app.post("/api/generate-text")
def generate_text(req: TextRequest):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=req.prompt,
        )

        text = response.text if hasattr(response, "text") else None

        if not text:
            return {"error": "No text generated"}

        return {"text": text}

    except Exception as e:
        return {"error": f"Text generation failed: {type(e).__name__}: {e}"}


# =========================
# IMAGE GENERATION
# =========================

@app.post("/api/generate-image")
def generate_image(req: ImageRequest):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=req.prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                data = part.inline_data.data

                if isinstance(data, (bytes, bytearray)):
                    data = base64.b64encode(data).decode("utf-8")

                return {
                    "image_base64": data,
                    "mime_type": part.inline_data.mime_type or "image/png",
                }

        return {"error": "Image not generated (no inline_data found)"}

    except Exception as e:
        return {"error": f"Image generation failed: {type(e).__name__}: {e}"}


# =========================
# STATIC FRONTEND
# =========================

app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/health")
def health():
    return {
        "ok": True,
        "has_key": bool(API_KEY),
        "text_model": TEXT_MODEL,
        "image_model": IMAGE_MODEL,
        "key_color": KEY_COLOR
    }


def _extract_inline_image(resp):
    """
    Returns: (base64_str, mime_type) or (None, None)
    """
    # Typical location: candidates[].content.parts[].inline_data
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data  # often already base64 string
                if isinstance(data, (bytes, bytearray)):
                    data = base64.b64encode(data).decode("utf-8")
                return data, getattr(inline, "mime_type", "image/png")

    # Fallback: resp.parts
    for part in (getattr(resp, "parts", None) or []):
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            if isinstance(data, (bytes, bytearray)):
                data = base64.b64encode(data).decode("utf-8")
            return data, getattr(inline, "mime_type", "image/png")

    return None, None


@app.post("/generate_text")
def generate_text(req: TextRequest):
    if client is None:
        return {"error": "GEMINI_API_KEY is not set in Render Environment Variables."}

    try:
        resp = client.models.generate_content(
            model=TEXT_MODEL,
            contents=[req.prompt],
        )

        text = getattr(resp, "text", None)
        if text:
            return {"text": text}

        parts = []
        for cand in (resp.candidates or []):
            for p in (cand.content.parts or []):
                if getattr(p, "text", None):
                    parts.append(p.text)
        return {"text": "\n".join(parts).strip() or ""}
    except Exception as e:
        return {"error": f"Text generation failed: {type(e).__name__}: {e}"}


@app.post("/generate_image")
def generate_image(req: ImageRequest):
    """
    Monolithic image generation (kept for compatibility).
    """
    if client is None:
        return {"error": "GEMINI_API_KEY is not set in Render Environment Variables."}

    try:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[req.prompt],
        )
        img_b64, mime = _extract_inline_image(resp)
        if img_b64:
            return {"image_base64": img_b64, "mime_type": mime}
        return {"error": "Image not generated (no inline_data in response). Try simplifying the prompt."}
    except Exception as e:
        return {"error": f"Image generation failed: {type(e).__name__}: {e}"}


@app.post("/generate_layer")
def generate_layer(req: LayerRequest):
    """
    Pseudo-layer generation:
    - background: normal prompt
    - object: enforce chroma-key background and "single subject" constraints
    """
    if client is None:
        return {"error": "GEMINI_API_KEY is not set in Render Environment Variables."}

    key_color = (req.key_color or KEY_COLOR).upper()

    base_prompt = (req.prompt or "").strip()
    if not base_prompt:
        return {"error": "Empty prompt."}

    if req.layer_kind.lower() == "background":
        final_prompt = (
            f"{base_prompt}\n\n"
            f"Rules:\n"
            f"- Background-only plate, no main subject in foreground.\n"
            f"- High quality, coherent lighting.\n"
            f"- No text, no watermark.\n"
        )
    else:
        # object layer with chroma key
        final_prompt = (
            f"{base_prompt}\n\n"
            f"STRICT RULES:\n"
            f"- ONE main object/character only.\n"
            f"- Place it centered, fully visible (not cropped).\n"
            f"- Background must be a perfectly solid flat color {key_color} (no gradient, no shadows on background).\n"
            f"- No additional objects, no scenery, no text, no watermark.\n"
            f"- Keep edges clean.\n"
        )

    try:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[final_prompt],
        )
        img_b64, mime = _extract_inline_image(resp)
        if img_b64:
            return {
                "image_base64": img_b64,
                "mime_type": mime,
                "key_color": key_color,
                "layer_kind": req.layer_kind,
                "layer_name": req.layer_name
            }
        return {"error": "Layer image not generated (no inline_data in response). Try simplifying the prompt."}
    except Exception as e:
        return {"error": f"Layer generation failed: {type(e).__name__}: {e}"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")def generate_text(req: TextRequest):
    if client is None:
        return {"error": "GEMINI_API_KEY is not set in Render Environment Variables."}

    try:
        resp = client.models.generate_content(
            model=TEXT_MODEL,
            contents=[req.prompt],
        )
        text = getattr(resp, "text", None)
        if text:
            return {"text": text}

        parts = []
        for cand in (resp.candidates or []):
            for p in (cand.content.parts or []):
                if getattr(p, "text", None):
                    parts.append(p.text)
        return {"text": "\n".join(parts).strip() or ""}
    except Exception as e:
        return {"error": f"Text generation failed: {type(e).__name__}: {e}"}

@app.post("/generate_image")
def generate_image(req: ImageRequest):
    if client is None:
        return {"error": "GEMINI_API_KEY is not set in Render Environment Variables."}

    try:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[req.prompt],
        )

        for cand in (resp.candidates or []):
            for part in (cand.content.parts or []):
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    data = inline.data  # usually base64 string
                    if isinstance(data, (bytes, bytearray)):
                        data = base64.b64encode(data).decode("utf-8")
                    return {"image_base64": data, "mime_type": getattr(inline, "mime_type", "image/png")}

        for part in (getattr(resp, "parts", []) or []):
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data
                if isinstance(data, (bytes, bytearray)):
                    data = base64.b64encode(data).decode("utf-8")
                return {"image_base64": data, "mime_type": getattr(inline, "mime_type", "image/png")}

        return {"error": "Image not generated (no inline_data in response). Try simplifying the prompt."}
    except Exception as e:
        return {"error": f"Image generation failed: {type(e).__name__}: {e}"}
       
        
