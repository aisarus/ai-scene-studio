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
if not API_KEY:
    client = None
else:
    client = genai.Client(api_key=API_KEY)

TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

class TextRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"ok": True, "has_key": bool(API_KEY), "text_model": TEXT_MODEL, "image_model": IMAGE_MODEL}

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
