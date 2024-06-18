# vectorDB service

RESTful API service of vector DB.

## Quickstart

```bash
export VECTORDB_SERVICE_HOST_IP=<ip of host where vector db is running>
docker compose build
docker compose up
```

## API request example

For full API docs, please visit `http://server_ip:9001/docs` for automatic interactive API documentation (provided by Swagger UI).

Template functions:

```python
import requests
import json
from typing import Optional

def get_data(api_url:str, query:dict):
    try:
        response = requests.get(api_url, query)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        return None

def post_data(api_url: str, body:dict):
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(api_url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        return None

def post_file(api_url: str, images, metadatas):
    try:
        response = requests.post(api_url, files=images, data=metadatas)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        return None
```

General CRUD operations:

```python
# health check
api_url = "http://172.16.186.168:9001/health"
results = get_data(api_url)

# add table/collection to db
api_url = "http://172.16.186.168:9001/add_table"
table = {
    "db_name": "chroma", # chroma or vdms
    "table": "test", # table name
    "vtype": "text" # text or image
}
results = post_data(api_url, table)

# add texts to vectorstore
api_url = "http://172.16.186.168:9001/add_texts"
text_content = [ "A", "content", "list" ]
metadata_list = [
    { "source": "A" },
    { "source": "content" },
    { "source": "list" },
]
results = post_data(api_url, {  "db_name": "chroma", 
                                "table": "test",
                                "texts": text_content, 
                                "metadatas": metadata_list})

# add available images to vectorstore
api_url = "http://172.16.186.168:9001/add_images"
uris = [ "/path/image1", "/path/image2" ]
metadata_list = [
    { "video": "video1.mp4" },
    { "video": "video2.mp4" }
]
results = post_data(api_url, {  "db_name": "chroma",
                                "table": "test",
                                "uris": uris,
                                "metadatas": metadata_list})

# query
api_url = "http://172.16.186.168:9001/search"
Q = 'man holding red basket'
query_dict = {
    "db_name": "chroma",
    "table": "test",
    "vtype": "text",
    "query": Q
}
results = get_data(api_url, query_dict)
```

Video-llama specific operations:

```python
# init vectorstore
api_url = "http://172.16.186.168:9001/visual_rag_retriever/init_db"
results = post_data(api_url, {"selected_db": "chroma"})

# add texts to vectorstore
api_url = "http://172.16.186.168:9001/visual_rag_retriever/add_texts"
text_content = [ "A", "content", "list" ]
metadata_list = [
    { "source": "A" },
    { "source": "content" },
    { "source": "list" },
]
results = post_data(api_url, {"texts": text_content, "metadatas": metadata_list})

# add available images to vectorstore
api_url = "http://172.16.186.168:9001/visual_rag_retriever/add_images"
uris = [ "/path/image1", "/path/image2" ]
metadata_list = [
    { "video": "video1.mp4" },
    { "video": "video2.mp4" }
]
results = post_data(api_url, {"uris": uris, "metadatas": metadata_list})

# upload images to vectorstore
api_url = 'http://172.16.186.168:9001/visual_rag_retriever/upload_images'
images = [
    ('images', ('op_1_0320241830.mp4_75.jpg', open('op_1_0320241830.mp4_75.jpg', 'rb'), 'image/jpeg')),
    ('images', ('op_1_0320241830.mp4_75.jpg', open('op_1_0320241830.mp4_75.jpg', 'rb'), 'image/jpeg'))
]
metadatas_dict = {"metadatas": [{"name": "John", "age": 30}, {"name": "Alice", "age": 25}]}
metadatas = {'metadatas': json.dumps(metadatas_dict)}
results = post_file(api_url, images, metadatas)

# multi modal query
api_url = "http://172.16.186.168:9001/visual_rag_retriever/query"
Q = 'man holding red basket'
results = get_data(api_url, {"prompt": Q})
```
