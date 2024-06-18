# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
# persistent flag
is_persistent = os.getenv("IS_PERSISTENT", "False")

# Read the timezone from /etc/timezone
timezone_cmd = ["cat", "/etc/timezone"]
timezone_result = subprocess.run(timezone_cmd, capture_output=True, text=True, check=True)

if timezone_result.returncode == 0:
    timezone = timezone_result.stdout.strip()

    # Create symbolic link to /etc/localtime
    ln_cmd = ["ln", "-snf", f"/usr/share/zoneinfo/{timezone}", "/etc/localtime"]
    ln_result = subprocess.run(ln_cmd, capture_output=True, check=True)

    if ln_result.returncode == 0:
        print("Timezone set successfully.")
    else:
        print("Error setting timezone.")
else:
    print("Error reading timezone file.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # init the db Handler on start up
    configs = utilities.read_config(CONFIG_PATH)
    app.state.db_handler = DB_Handler(configs)

    # Load the objects from the pickle file
    if is_persistent == "True":
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
