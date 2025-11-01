# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import analyze_error

app = FastAPI(title="DevOps Troubleshooter")

class ErrorRequest(BaseModel):
    error: str

@app.post("/analyze-log")
async def analyze_log(req: ErrorRequest):
    if not req.error or len(req.error) < 3:
        raise HTTPException(status_code=400, detail="Provide an error string in 'error' field")
    try:
        result = analyze_error(req.error)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

