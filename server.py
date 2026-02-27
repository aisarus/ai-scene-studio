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
