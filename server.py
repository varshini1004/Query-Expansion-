from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from query_expander import NeuralExpander
import asyncio

app = FastAPI()
expander = NeuralExpander()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    await expander.init_session()

@app.on_event("shutdown")
async def shutdown_event():
    await expander.close()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class QueryRequest(BaseModel):
    query: str

@app.post("/expand")
async def expand_query(req: QueryRequest):
    result = await expander.expand_query(req.query)
    return result