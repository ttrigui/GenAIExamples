import sys
import os
from tqdm import tqdm

# Add the parent directory of the current script to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
VECTORDB_SERVICE_HOST_IP = os.getenv("VECTORDB_SERVICE_HOST_IP", "0.0.0.0")


# sys.path.append(os.path.abspath('../utils'))
# import config_reader as reader
import yaml
import json
import os
import argparse
import torch
from embedding.vclip.vclip import vCLIP
from utils import config_reader as reader
from embedding.extract_store_frames import process_all_videos
from embedding.vector_stores import db


def setup_vclip_model(config, device="cpu"):
    
    model = vCLIP(config)

    return model


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
    
    for idx, (video, data) in enumerate(tqdm(GMetadata.items())):
        image_name_list = []
        embedding_list = []
        metadata_list = []
        ids = []
        
        if config['embeddings']['type'] == 'video':
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
                print(f"ERROR: selected_db {vs.selected_db} not supported. Supported:[vdms]")

    # clean up tmp_ folders containing frames (jpeg)
    for i in os.listdir():
        if i.startswith("tmp_"):
            print("removing tmp_*")
            os.system(f"rm -r tmp_*")
            print("done.")
            break

def generate_embeddings(config, embedding_model, vs):
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
    # Read vCLIP
    meanclip_cfg = {"model_name": config['embeddings']['vclip_model_name'], "num_frm": config['embeddings']['vclip_num_frm']}

    print ('Config file data \n', yaml.dump(config, default_flow_style=False, sort_keys=False))

    generate_frames = config['generate_frames']
    #embed_frames = config['embed_frames']
    path = config['videos'] #args.videos_folder #
    meta_output_dir = config['meta_output_dir']
    emb_path = config['embeddings']['path']

    host = VECTORDB_SERVICE_HOST_IP
    port = int(config['vector_db']['port'])
    selected_db = config['vector_db']['choice_of_db']

    # Creating DB
    print ('Creating DB with video embedding and metadata support, \nIt may take few minutes to download and load all required models if you are running for first time.')
    print('Connecting to {} at {}:{}'.format(selected_db, host, port))

    if config['embeddings']['type'] == 'video':
        # init meanclip model
        model = setup_vclip_model(meanclip_cfg, device="cpu")
        vs = db.VideoVS(host, port, selected_db, model)
    else:
        print(f"ERROR: Selected embedding type in config.yaml {config['embeddings']['type']} is not in [\'video\', \'frame\']")
        return
    generate_embeddings(config, model, vs)
    retrieval_testing(vs)
    return vs

if __name__ == '__main__':
    main()