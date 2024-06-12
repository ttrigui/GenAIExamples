from contextlib import asynccontextmanager
import os
import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils import utilities
import uvicorn
from routers import (
    health,
    crud,
    video_llama
)
from core.db_handler import DB_Handler

CONFIG_PATH = "/gateway/conf/conf.yaml"

# align /etc/timezone and /etc/localtime
command = "TZ=`cat /etc/timezone` && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime"
result = subprocess.run(command, shell=True, capture_output=True, text=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # init the db Handler on start up
    configs = utilities.read_config(CONFIG_PATH)
    app.state.db_handler = DB_Handler(configs)

    # Load the objects from the pickle file
    handler_pickle_path = configs['handler_pickle_path']
    if os.path.exists(handler_pickle_path):
        app.state.db_handler.load_from_pkl_file(handler_pickle_path)
    
    yield

app = FastAPI(
    title="VectorDB retriever API",
    description="",
    redoc_url="/",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(crud.router)
app.include_router(video_llama.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=True)
