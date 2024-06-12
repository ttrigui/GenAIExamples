import json
import logging
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from core.db_handler import DB_Handler
from utils.utilities import get_db_handler

VIDEO_LLAMA_TEXT_COLLECTION = 'text-test'
VIDEO_LLAMA_IMAGE_COLLECTION = 'image-test'
UPLOAD_FRAMES_FOLDER = "/home/data/frames"

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     [%(asctime)s] %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S'
    )

router = APIRouter(prefix="/visual_rag_retriever", tags=["visual_rag_retriever"])

class DB_Name(BaseModel):
    selected_db: Optional[str] = Field(..., description="Supported: chroma, vdms", example="chroma")

class ImageMetadata(BaseModel):
    metadatas: List[dict]

    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

@router.post("/init_db", summary="Init db for video llama")
@router.post("/init_db/", include_in_schema=False)
def init_db(
        db_name: DB_Name,
        db_handler: DB_Handler = Depends(get_db_handler)
    ):

    try:
        logging.info('Loading db instances')
        db_handler.selected_db = db_name.selected_db # update to the whole class
       
        # db_handler.add_table(selected_db, VIDEO_LLAMA_TEXT_COLLECTION, 'text') # visual RAG updated: only image db
        db_handler.add_table(db_handler.selected_db, VIDEO_LLAMA_IMAGE_COLLECTION, 'image')
    except Exception as e:
        logging.error(f"Init table failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, 
                            detail=f"Failed to init table, please check the params and try again: {e}") from e

    return JSONResponse({"message:": f"successfully init {db_name.selected_db} for video-llama"})
  
@router.post("/add_texts", summary="Add texts to video llama text db")
@router.post("/add_texts/", include_in_schema=False)
async def add_texts(
        texts: List[str], 
        metadatas: Optional[List[dict]] = None, 
        db_handler: DB_Handler = Depends(get_db_handler)
    ):

    try:
        db_handler.add_texts(db_handler.selected_db, VIDEO_LLAMA_TEXT_COLLECTION, texts, metadatas)
    except Exception as e:
        logging.error(f"Add texts failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, 
                            detail=f"Failed to add texts, please check the params and try again: {e}") from e
    
    return JSONResponse({"message:": f"successfully add {len(texts)} texts to {db_handler.selected_db}"})

@router.post("/add_images", summary="Add images to video llama image db")
@router.post("/add_images/", include_in_schema=False)
async def add_images(
        uris: List[str], 
        metadatas: Optional[List[dict]] = None, 
        db_handler: DB_Handler = Depends(get_db_handler)
    ):

    try:
        db_handler.add_images(db_handler.selected_db, VIDEO_LLAMA_IMAGE_COLLECTION, uris, metadatas)
    except Exception as e:
        logging.error(f"Add images failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, 
                            detail=f"Failed to add images, please check the params and try again: {e}") from e

    return JSONResponse({"message:": f"successfully add {len(uris)} images to {db_handler.selected_db}"})

@router.get("/query", summary="Query video llama retriever, multi-modal")
@router.get("/query/", include_in_schema=False)
async def visual_rag_query(
        prompt: str,
        db_handler: DB_Handler = Depends(get_db_handler)
    ):

    try:
        results = db_handler.visual_rag_retrieval(VIDEO_LLAMA_IMAGE_COLLECTION, prompt)
        # empty page content for better latency
        for item in results:
            item.page_content = ""
    except Exception as e:
        logging.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=400,
                            detail=f"Failed to query, please check the params and try again: {e}") from e

    return results 

@router.get("/single_modal_query", summary="Query video llama retriever, common single-modal")
@router.get("/single_modal_query/", include_in_schema=False)
async def single_modal_query(
        prompt: str,
        vtype: str,
        n_results: int = 1,
        db_handler: DB_Handler = Depends(get_db_handler)
    ):
    if vtype == "text":
        collection = VIDEO_LLAMA_TEXT_COLLECTION
    elif vtype == "image":
        collection = VIDEO_LLAMA_IMAGE_COLLECTION
    else:
        raise HTTPException(status_code=400,
                            detail="Failed to query, supported types are 'text' and 'image'")
    try:
        results = db_handler.single_modal_retrieval(db_handler.selected_db, collection, vtype, prompt, n_results)
        # empty page content for better latency
        for item in results:
            item.page_content = ""
    except Exception as e:
        logging.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=400,
                            detail=f"Failed to query, please check the params and try again: {e}") from e

    return results 

@router.post("/upload_images", summary="Upload image and add it to image vector DB")
@router.post("/upload_images/", include_in_schema=False)
async def upload_images(
        images: List[UploadFile],
        metadatas: ImageMetadata,
        db_handler: DB_Handler = Depends(get_db_handler)
    ):

    """
    accept image files and metadata, save them to files folder and call add_images \n
    Args: \n
        images: image list \n
        metadata: object with a list of dict as single field \n
    Returns: \n
        JSONResponse: including images file path and success status.
    """
    saved_paths = []
    os.makedirs(UPLOAD_FRAMES_FOLDER, exist_ok=True)
    
    for image in images:
        contents = await image.read()
        image_name = image.filename
        image_path = os.path.join(UPLOAD_FRAMES_FOLDER, image_name)
        with open(image_path, "wb") as f:
            f.write(contents)
        saved_paths.append(image_path)

    try:
        db_handler.add_images(db_handler.selected_db, VIDEO_LLAMA_IMAGE_COLLECTION, saved_paths, metadatas.metadatas)
    except Exception as e:
        logging.error(f"Add images failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Failed to add images") from e

    return JSONResponse({"saved_paths": saved_paths})
