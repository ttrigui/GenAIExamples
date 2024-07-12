# import sys
# import os

# sys.path.append('/path/to/parent')  # Replace with the actual path to the parent folder

# from VideoRAGQnA.utils import config_reader as reader
import sys
import os

# Add the parent directory of the current script to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
VECTORDB_SERVICE_HOST_IP = os.getenv("VECTORDB_SERVICE_HOST_IP", "0.0.0.0")


# sys.path.append(os.path.abspath('../utils'))
# import config_reader as reader
import yaml
import chromadb
import json
import os
import argparse
import torch
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from embedding.meanclip_modeling.model import MeanCLIP
from embedding.meanclip_modeling.clip_model import CLIP
from utils import config_reader as reader
from embedding.extract_store_frames import process_all_videos
from embedding.vector_stores import db
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
from embedding.meanclip_datasets.preprocess import get_transforms


def setup_meanclip_model(cfg, device):

    pretrained_state_dict = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    state_dict = {}
    epoch = 0
    print("Loading CLIP pretrained weights ...")
    for key, val in pretrained_state_dict.items():    
        new_key = "clip." + key
        if new_key not in state_dict:
            state_dict[new_key] = val.clone()

    if cfg.sim_header != "meanP":
        for key, val in pretrained_state_dict.items():
            # initialize for the frame and type postion embedding
            if key == "positional_embedding":
                state_dict["frame_position_embeddings.weight"] = val.clone()

            # using weight of first 4 layers for initialization
            if key.find("transformer.resblocks") == 0:
                num_layer = int(key.split(".")[2])

                # initialize the 4-layer temporal transformer
                if num_layer < 4:
                    state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                    continue

                if num_layer == 4: # for 1-layer transformer sim_header
                    state_dict[key.replace(str(num_layer), "0")] = val.clone()

    model = MeanCLIP(cfg, pretrained_state_dict)
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if str(device) == "cpu":
        model.float()

    model.to(device)

    print("Setup model done!")
    return model, epoch

def read_json(path):
    with open(path) as f:
        x = json.load(f)
    return x

def read_file(path):
    content = None
    with open(path, 'r') as file:
        content = file.read()
    return content

def store_into_vectordb(vs, metadata_file_path, embedding_model, config):
    GMetadata = read_json(metadata_file_path)
    global_counter = 0

    total_videos = len(GMetadata.keys())
    
    for idx, (video, data) in enumerate(GMetadata.items()):

        image_name_list = []
        embedding_list = []
        metadata_list = []
        ids = []
        
        if config['embeddings']['type'] == 'frame':
            # process frames
            frame_metadata = read_json(data['extracted_frame_metadata_file'])
            for frame_id, frame_details in frame_metadata.items():
                global_counter += 1
                if vs.selected_db == 'vdms':
                    meta_data = {
                        'timestamp': frame_details['timestamp'],
                        'frame_path': frame_details['frame_path'],
                        'video': video,
                        #'embedding_path': data['embedding_path'],
                        'date_time': frame_details['date_time'], #{"_date":frame_details['date_time']},
                        'date': frame_details['date'],
                        'year': frame_details['year'],
                        'month': frame_details['month'],
                        'day': frame_details['day'],
                        'time': frame_details['time'],
                        'hours': frame_details['hours'],
                        'minutes': frame_details['minutes'],
                        'seconds': frame_details['seconds'],
                    }
                if vs.selected_db == 'chroma':
                    meta_data = {
                        'timestamp': frame_details['timestamp'],
                        'frame_path': frame_details['frame_path'],
                        'video': video,
                        #'embedding_path': data['embedding_path'],
                        'date': frame_details['date'],
                        'year': frame_details['year'],
                        'month': frame_details['month'],
                        'day': frame_details['day'],
                        'time': frame_details['time'],
                        'hours': frame_details['hours'],
                        'minutes': frame_details['minutes'],
                        'seconds': frame_details['seconds'],
                        'hnsw:space': "ip", # Inner Product as distance_key for similarity
                    }
                image_path = frame_details['frame_path']
                image_name_list.append(image_path)

                metadata_list.append(meta_data)
                ids.append(str(global_counter))
                # print('datetime',meta_data['date_time'])

            vs.add_images(
                uris=image_name_list,
                metadatas=metadata_list
            )
        elif config['embeddings']['type'] == 'video':
            data['video'] = video
            video_name_list = [data["video_path"]]
            metadata_list = [data]
            if vs.selected_db == 'vdms':
                vs.video_db.add_videos(
                    paths=video_name_list,
                    metadatas=metadata_list,
                    start_time=[data['timestamp']],
                    clip_duration=[data['clip_duration']]
                )
            else:
                # Call local embedding function to retrieve tensor
                tensor = vs.video_embedder.embed_video(paths=video_name_list,
                    metadatas=metadata_list,
                    start_time=[data['timestamp']],
                    clip_duration=[data['clip_duration']])
                vs.video_db._collection.add(
                    metadatas=metadata_list,
                    embeddings=tensor,
                    ids=[f"{idx}"]
                )
        print (f'✅ {idx+1}/{total_videos} video {video}')

def generate_embeddings(config, embedding_model, vs):
    if not os.path.exists(config['image_output_dir']):
        print ('Processing all videos, Generated frames will be stored at')
        print (f'input video folder = {config["videos"]}')
        print (f'frames output folder = {config["image_output_dir"]}')
        print (f'metadata files output folder = {config["meta_output_dir"]}')
        process_all_videos(config)
    global_metadata_file_path = os.path.join(config["meta_output_dir"], 'metadata.json')
    print(f'global metadata file available at {global_metadata_file_path}')
    store_into_vectordb(vs, global_metadata_file_path, embedding_model, config)

def retrieval_testing(vs):
    Q = 'Man holding red shopping basket'
    print (f'Testing Query: {Q}')
    top_k = 3
    results = vs.MultiModalRetrieval(Q, top_k=top_k)

    print(f"top-{top_k} returned results:", results)

def main():
    # read config yaml
    print ('Reading config file')
    # config = reader.read_config('../docs/config.yaml')

    # Create argument parser
    parser = argparse.ArgumentParser(description='Process configuration file for generating and storing embeddings.')
    parser.add_argument('config_file', type=str, help='Path to configuration file (e.g., config.yaml)')

    # Parse command-line arguments
    args = parser.parse_args()
    # Read configuration file
    config = reader.read_config(args.config_file)
    # Read MeanCLIP
    meanclip_cfg_json = json.load(open(config['meanclip_cfg_path'], 'r'))
    meanclip_cfg = argparse.Namespace(**meanclip_cfg_json)


    print ('Config file data \n', yaml.dump(config, default_flow_style=False, sort_keys=False))

    generate_frames = config['generate_frames']
    #embed_frames = config['embed_frames']
    path = config['videos'] #args.videos_folder #
    image_output_dir = config['image_output_dir']
    meta_output_dir = config['meta_output_dir']
    N = config['number_of_frames_per_second']
    emb_path = config['embeddings']['path']

    host = VECTORDB_SERVICE_HOST_IP
    port = int(config['vector_db']['port'])
    selected_db = config['vector_db']['choice_of_db']

    # Creating DB
    print ('Creating DB with video embedding and metadata support, \nIt may take few minutes to download and load all required models if you are running for first time.')
    print('Connecting to {} at {}:{}'.format(selected_db, host, port))

    if config['embeddings']['type'] == 'frame':
        vs = db.VS(host, port, selected_db)
        # EMBEDDING MODEL
        model = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")

    elif config['embeddings']['type'] == 'video':
        # init meanclip model
        model, _ = setup_meanclip_model(meanclip_cfg, device="cpu")
        vs = db.VideoVS(host, port, selected_db, model)
    else:
        print(f"ERROR: Selected embedding type in config.yaml {config['embeddings']['type']} is not in [\'video\', \'frame\']")
        return
    generate_embeddings(config, model, vs)
    retrieval_testing(vs)
    return vs

if __name__ == '__main__':
    main()
