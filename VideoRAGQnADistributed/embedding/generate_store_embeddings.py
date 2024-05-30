from typing import Optional
import yaml
import json
import os
from extract_store_frames import process_all_videos
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import requests
import base64

print ('Reading config file')
config = None
with open("../docs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# EMBEDDING MODEL
clip_embd = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k") 
 
def read_json(path):
    with open(path) as f:
        x = json.load(f)
    return x
 
def read_file(path):
    content = None
    with open(path, 'r') as file:
        content = file.read()
    return content
 
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
 
def get_data(api_url:str, query:Optional[dict] = None):
    try:
        response = requests.get(api_url, query)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        return None
 
def encode_image(uri: str) -> str:
    """Get base64 string from image URI."""
    with open(uri, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
 
 
def init_vectordb(db_kind:str):
    url = config["vector_init_url"]
    results = post_data(url, {"selected_db": db_kind})
    return results

def check_health_vectordb():
    url = config["vector_health_url"]
    response = get_data(url, {})
    return response

def store_into_vectordb(metadata_file_path):
    print("metadata_file_path", metadata_file_path)
    GMetadata = read_json(metadata_file_path)
    global_counter = 0
 
    total_videos = len(GMetadata.keys())
    image_insert_url = config["image_insert_url"]
    for _, (video, data) in enumerate(GMetadata.items()):
 
        image_name_list = []
        image_files = []
        embedding_list = []
        metadata_list = []
        ids = []
        # process frames
        frame_metadata = read_json(data['extracted_frame_metadata_file'])
        for frame_id, frame_details in frame_metadata.items():
            global_counter += 1
            meta_data = {
                'timestamp': frame_details['timestamp'],
                'frame_path': frame_details['frame_path'],
                'video': video,
                'embedding_path': data['embedding_path'],
            }
            image_path = frame_details['frame_path']
            image_name_list.append(image_path)
            
            metadata_list.append(meta_data)
            ids.append(str(global_counter))
 
        # generate clip embeddings
        embedding_list.extend(clip_embd.embed_image(image_name_list))
 
        for image_path in image_name_list:
            with open(image_path, 'rb') as image_file:
                image_files.append(('images', (os.path.basename(image_path), image_file.read(), 'image/jpeg')))
        
        metadata_dict = {"metadatas": metadata_list}
        metadatas = {'metadatas': json.dumps(metadata_dict)}
        results = post_file(image_insert_url, image_files, metadatas)

        print (f'✅ {_+1}/{total_videos} video {video}, len {len(image_name_list)}, {len(metadata_list)}, {len(embedding_list)}')
 
def generate_image_embeddings():
    if generate_frames:
        print ('Processing all videos, Generated frames will be stored at')
        print (f'input video folder = {path}')
        print (f'frames output folder = {image_output_dir}')
        print (f'metadata files output folder = {meta_output_dir}')
        process_all_videos(path, image_output_dir, meta_output_dir, N)
    global_metadata_file_path = meta_output_dir + 'metadata.json'
    print(f'global metadata file available at {global_metadata_file_path}')
    store_into_vectordb(global_metadata_file_path)

def generate_text_embeddings():
    all_videos = os.listdir(path)
    # each scene description is a document in vector storage
    text_content = []
    metadata_list = []
    for video in all_videos:
        description_path = os.path.join(config['description'], video + '.txt')
        if os.path.exists(description_path):
            # read file content and prepare document 
            text = read_file(description_path)
            text_content.append(text)
            # video == video name
            metadata =  {
                    'video': video # video path
                }
            metadata_list.append(metadata)
    text_insert_url = config["text_insert_url"]
    results = post_data(text_insert_url, {"texts": text_content, "metadatas": metadata_list})

def retrieval_testing():
    Q = 'man holding red basket'
    print (f'Testing Query {Q}')
    url = config["vector_query_url"]
    results = get_data(url, {"prompt": Q})
    print (results)

if __name__ == '__main__':
    print ('Config file data \n', yaml.dump(config, default_flow_style=False, sort_keys=False))
 
    generate_frames = config['generate_frames']
    embed_frames = config['embed_frames']
    path = "../"+config['videos']
    image_output_dir = "../"+config['image_output_dir']
    meta_output_dir = "../"+config['meta_output_dir']
    N = config['number_of_frames_per_second']
    selected_db = config['vector_db']['choice_of_db']
    
    # Creating DB
    print ('Creating DB with text and image embedding support, \nIt may take few minutes to download and load all required models if you are running for first time.')
    check_health_vectordb() # health check for vectorDB
    init_vectordb(selected_db)

    generate_text_embeddings()
    retrieval_testing()
    generate_image_embeddings()
    retrieval_testing()