from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import base64

# Official Google Gen AI SDK (Gemini API)
from google import genai

app = FastAPI()

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

KEY_COLOR = "#00FF00"  # chroma key for pseudo-layers

class TextRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    prompt: str

class LayerRequest(BaseModel):
    prompt: str
    layer_name: str
    layer_kind: str  # "background" | "object"
    key_color: str | None = None  # optional override


@app.get("/")
def root():
    return FileResponse("static/index.html")


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
        app.mount("/", StaticFiles(directory="static", html=True), name="static")
