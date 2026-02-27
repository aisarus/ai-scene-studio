# Complete cleaned-up version of server.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles  # Added missing import

app = FastAPI()

# Define your routes and business logic
@app.get("/")
async def read_root():
    return {"Hello": "World"}

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# More routes and functionality can go here...
