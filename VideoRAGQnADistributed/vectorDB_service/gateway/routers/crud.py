import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from core.db_handler import DB_Handler
from utils import utilities

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:    [%(asctime)s] %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S'
    )

router = APIRouter(tags=["Basic CRUD operations"])

def get_db_handler():
    from main import db_handler
    return db_handler

class Table(BaseModel):
    db_name: str = Field(..., description="Supported: chroma, vdms", example="chroma")
    table: str = Field(..., description="table or collection name", example="test", min_length=3, max_length=63)
    vtype: str = Field(..., description="Supported: text, image", example="text")

class Texts(BaseModel):
    db_name: str = Field(..., description="Supported: chroma, vdms", example="chroma")
    table: str = Field(..., description="table or collection name", example="test")
    texts: List[str] = Field(..., description="list of text contents", example=["text1", "text2"])
    metadatas: Optional[List[dict]] = Field(None,
                                            description="list of metadata",
                                            example=[{"video": "video1.mp4"}, {"video": "video2.mp4"}])

class Images(BaseModel):
    db_name: str = Field(..., description="Supported: chroma, vdms", example="chroma")
    table: str = Field(..., description="table or collection name", example="test")
    uris: List[str] = Field(..., description="list of image URIs", example=["uri1", "uri2"])
    metadatas: Optional[List[dict]] = Field(None,
                                            description="list of metadata",
                                            example=[{"video": "video1.mp4"}, {"video": "video2.mp4"}])

class Record(BaseModel):
    vtype: str = Field(..., description="Supported: text, image", example="text")
    texts: Optional[Texts] = None
    images: Optional[Images] = None

@router.post("/add_table", summary="add_table")
@router.post("/add_table/", include_in_schema=False)
async def add_table(
    new_table: Table,
    db_handler: DB_Handler = Depends(get_db_handler)
    ):

    """
    db_name: chroma, vdms\n
    table: table or collection name\n
    vtype: text or image, for embedder choosing
    """

    try:
        db_handler.add_table(new_table.db_name, new_table.table, new_table.vtype)
    except Exception as e:
        logging.error(f"Add table failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, 
                            detail=f"Failed to add table, please check the params and try again: {e}") from e

    return JSONResponse({"message:": f"successfully add {new_table.table} to {new_table.db_name}"})

@router.post("/add_texts", summary="add_texts")
@router.post("/add_texts/", include_in_schema=False)
async def add_texts(
    new_text: Texts, 
    db_handler: DB_Handler = Depends(get_db_handler)
    ):

    """
    db_name: chroma, vdms\n
    table: table or collection name\n
    texts: list of text contents\n
    metadatas: list of metadata, e.g. [{ "video": "video_name.mp4" }]
    """

    logging.info("Received texts: %s", new_text.texts)
    logging.info("Received metadatas: %s", new_text.metadatas)
    try:
        db_handler.add_texts(new_text.db_name, new_text.table, new_text.texts, new_text.metadatas)
    except Exception as e:
        logging.error(f"Add texts failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, 
                            detail=f"Failed to add texts, please check the params and try again: {e}") from e

    return JSONResponse({"message:": f"successfully add {len(new_text.texts)} texts to {new_text.table} in {new_text.db_name}"})

@router.post("/add_images", summary="add_images")
@router.post("/add_images/", include_in_schema=False)
async def add_images(
    new_image: Images,
    db_handler: DB_Handler = Depends(get_db_handler)
    ):

    """
    db_name: chroma, vdms\n
    table: table or collection name\n
    uris: list of image URIs\n
    metadatas: list of metadata, e.g. [{ "timestamp": "123", "frame_path": "uri", "video": "video_name.mp4", "embedding_path": "xxx.pt" }]
    """

    logging.info("Received uris: %s", new_image.uris)
    logging.info("Received metadatas: %s", new_image.metadatas)
    try:
        db_handler.add_images(new_image.db_name, new_image.table, new_image.uris, new_image.metadatas)
    except Exception as e:
        logging.error(f"Add images failed: {e}", exc_info=True)
        raise HTTPException(status_code=400,
                            detail=f"Failed to add images, please check the params and try again: {e}") from e

    return JSONResponse({"message:": f"successfully add {len(new_image.uris)} images to {new_image.table} in {new_image.db_name}"})


@router.post("/update", summary="update")
@router.post("/update/", include_in_schema=False)
async def update(
    record: Record, 
    db_handler: DB_Handler = Depends(get_db_handler)
    ):

    try:
        if record.vtype == "text":
            db_handler.add_texts(record.texts.db_name, record.texts.table, record.texts.texts, record.texts.metadatas)
        elif record.vtype == "image":
            db_handler.add_images(record.images.db_name, record.images.table, record.images.uris, record.images.metadatas)
    except Exception as e:
        logging.error(f"Update failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, 
                            detail=f"Failed to update, please check the params and try again: {e}") from e

    return JSONResponse({"message:": f"successfully update {record.vtype} to {record.texts.table} in {record.texts.db_name}"})

@router.get("/search", summary="Search database for a query")
@router.get("/search/", include_in_schema=False)
async def search(
    db_name: str = Query(..., description="Supported: chroma, vdms", example="chroma"), 
    table: str = Query(..., description="table or collection name", example="test"),
    vtype: str = Query(..., description="Supported: text, image", example="text"),
    query: str = Query(..., description="The query to search for.", example="hello"),
    n_results: int = Query(default=1, description="The number of results to return."),
    db_handler: DB_Handler = Depends(get_db_handler)
    ):

    """
    Returns: List[dict]: A list of dictionaries, each containing a document matched by the query.
    """
    try:
        results = db_handler.single_modal_retrieval(db_name, table, vtype, query, n_results)
    except Exception as e:
        logging.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=400,
                            detail=f"Failed to search, please check the params and try again: {e}") from e

    return results

@router.delete("/drop_table/{table}", summary="drop_table")
def drop_table(
    table: str,
    db_name: str,
    db_handler: DB_Handler = Depends(get_db_handler)
    ):

    """
    db_name: chroma, vdms\n
    table: table or collection name
    """
    try:
        db_handler.delete_collection(db_name, table)
    except Exception as e:
        logging.error(f"Drop table failed: {e}", exc_info=True)
        raise HTTPException(status_code=400,
                            detail=f"Failed to drop table, please check the params and try again: {e}") from e

    return JSONResponse({"message:": f"successfully drop {table} from {db_name}"})

@router.delete("/delete", summary="delete")
@router.delete("/delete/", include_in_schema=False)
async def delete(payload):
    pass
