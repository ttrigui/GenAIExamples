
#This file needs to read the long retail video, and for every thirty seconds, choose a 10 second interval to create embeddings. Store embeddings with metadata.

# from VideoRAGQnA.utils import config_reader as reader
import sys
import os

# Add the parent directory of the current script to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
VECTORDB_SERVICE_HOST_IP = os.getenv("VECTORDB_SERVICE_HOST_IP", "0.0.0.0")

import yaml
import chromadb
import json
import os
import argparse
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from utils import config_reader as reader
from embedding.vector_stores import db
import cv2
import random
import datetime
from tzlocal import get_localzone


def read_json(path):
    with open(path) as f:
        x = json.load(f)
    return x

# EMBEDDING MODEL
clip_embd = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")

# Have to change function to take in start time/frame and end time/frame 
def extract_frames(video_path, start_time, end_time, interval_count, date_time, local_timezone, meta_output_dir, image_output_dir, N=100, selected_db='chroma'):
        # video = video_path.split('/')[-1]

        video = os.path.basename(video_path)
        video, _ = os.path.splitext(video)
        # Create a directory to store frames and metadata
        image_output_dir = os.path.join(image_output_dir, f'{video}', 'interval_' + f'{interval_count}')
        print(image_output_dir)
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(meta_output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if int(cv2.__version__.split('.')[0]) < 3:
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
    
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        interval_frames = end_frame - start_frame

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #print (f'fps {fps}')
        #print (f'total frames {total_frames}')
        
        mod = int(fps // N)
        if mod == 0: mod = 1
        
        print (f'total frames in interval {interval_frames}, N {N}, mod {mod}')
        
        # Metadata dictionary to store timestamp and image paths
        metadata = {}

        # Move video to start time
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        # Variables to track frame count and desired frames
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

            if pos_msec >= end_time * 1000:
                break
            
            frame_count += 1

            # image_output_dir = image_output_dir + f'{video} + f{interval_counter}'
            # print('frame_path: ', os.path.join(image_output_dir, f"{video}_{frame_count}.jpg"))
            
            if frame_count % mod == 0:
                timestamp = start_time
                frame_path = os.path.join(image_output_dir, f"{video}_{frame_count}.jpg")
                time = date_time.strftime("%H:%M:%S")
                date = date_time.strftime("%Y-%m-%d")
                hours, minutes, seconds = map(float, time.split(":"))
                year, month, day = map(int, date.split("-"))
                
                cv2.imwrite(frame_path, frame)  # Save the frame as an image


                metadata[frame_count] = {"timestamp": timestamp, "frame_path": frame_path,"date": date, "year": year, "month": month, "day": day, 
                    "time": time, "hours": hours, "minutes": minutes, "seconds": seconds, "video": video_path}
                if selected_db == 'vdms':
                    # Localize the current time to the local timezone of the machine
                    #Tahani might not need this
                    current_time_local = date_time.replace(tzinfo=datetime.timezone.utc).astimezone(local_timezone)

                    # Convert the localized time to ISO 8601 format with timezone offset
                    iso_date_time = current_time_local.isoformat()
                    metadata[frame_count]['date_time'] = {"_date": str(iso_date_time)}

        # Save metadata to a JSON file
        # metadata_file = os.path.join(meta_output_dir, f"{video}_metadata.json")
        # with open(metadata_file, "w") as f:
        #     json.dump(metadata, f, indent=4)
        
        # Release the video capture and close all windows
        cap.release()
        print(f"{frame_count/mod} Frames extracted and metadata saved successfully.") 
        return fps, interval_frames, metadata

# This function needs to change arguemtns so it doesn't take videos, but frames or intervals to generate embeddings of intervals. Change functionality of this function to not need to save videos before generating embeddings, as well as do the embedding one at a time.
def store_into_vectordb(metadata_dict, selected_db):
    global_frame_counter = 0

    image_name_list = []
    embedding_list = []
    metadata_list = []
    ids = []
        
    # process frames
    for frame_id, frame_details in metadata_dict.items():
        global_frame_counter += 1
        if selected_db == 'vdms':
            meta_data = {
                'start of interval in sec': frame_details['timestamp'],
                'frame_path': frame_details['frame_path'],
		'video': frame_details['video'],
                # 'embedding_path': curr_data['embedding_path'],
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
        if selected_db == 'chroma':
            meta_data = {
                'start of interval in sec': frame_details['timestamp'],
                'frame_path': frame_details['frame_path'],
		'video': frame_details['video'],
                # 'embedding_path': curr_data['embedding_path'],
                'date': frame_details['date'],
                'year': frame_details['year'],
                'month': frame_details['month'],
                'day': frame_details['day'],
                'time': frame_details['time'],
                'hours': frame_details['hours'],
                'minutes': frame_details['minutes'],
                'seconds': frame_details['seconds'],
            }
        image_path = frame_details['frame_path']
        image_name_list.append(image_path)

        metadata_list.append(meta_data)
        ids.append(str(global_frame_counter))
        # print('datetime',meta_data['date_time'])
    # generate clip embeddings
    # embedding_list.extend(clip_embd.embed_image(image_name_list))

    vs.add_images(
        uris=image_name_list,
        metadatas=metadata_list
    )

    print("✅ Finished creating embeddings for interval")
    
    # print (f'✅ {_+1}/{total_videos} video {video}, len {len(image_name_list)}, {len(metadata_list)}, {len(embedding_list)}')

def calculate_intervals(video_path, chunk_duration, clip_duration):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / fps

    intervals = []

    chunk_frames = int(chunk_duration * fps)
    clip_frames = int(clip_duration * fps)

    for start_frame in range(0, total_frames, chunk_frames):
        end_frame = min(start_frame + clip_frames, total_frames)
        start_time = start_frame / fps
        end_time = end_frame / fps
        intervals.append((start_frame, end_frame, start_time, end_time))
    
    cap.release()
    return intervals

global interval_counter
interval_counter = 0
def process_video(video_path, selected_db, chunk_duration, clip_duration):
    interval_count = 0
    intervals = calculate_intervals(video_path, chunk_duration, clip_duration)

    for interval in intervals:
        start_frame, end_frame, start_time, end_time = interval

        date_time = datetime.datetime.now() 

        local_timezone = get_localzone()
        # With this interval, extract frames to create metadata
        fps, interval_frames, metadata_dict = extract_frames(video_path, start_time, end_time, interval_count, date_time, local_timezone, meta_output_dir, image_output_dir, N=100, selected_db='chroma')

        video = os.path.basename(video_path)
        video, _ = os.path.splitext(video)

        first_frame_id, first_frame_details = list(metadata_dict.items())[0]
        frames_path = os.path.dirname(first_frame_details['frame_path'])

        metadata = {}
        global_metadata_file = meta_output_dir + 'metadata.json'

        metadata[video + '_interval_' + f"{interval_count}"] = {
            "start datetime":
            {
                'start of interval in sec': first_frame_details['timestamp'],
                'date': first_frame_details['date'],
                'year': first_frame_details['year'],
                'month': first_frame_details['month'],
                'day': first_frame_details['day'],
                'time': first_frame_details['time'],
                'hours': first_frame_details['hours'],
                'minutes': first_frame_details['minutes'],
                'seconds': first_frame_details['seconds'],
            },
            "fps": fps,
            "total_frames": interval_frames,
            "embedding_path": f"embeddings/{video}.pt",
            "video_path": f"{video_path}",
            "frames_path": frames_path
        }

        with open(global_metadata_file, "a") as f:
            json.dump(metadata, f, indent=4)
        print("DICTIONARY USED FOR EMBEDDING: ", metadata_dict)
        store_into_vectordb(metadata_dict, selected_db='chroma')
        interval_count += 1

if __name__ == "__main__":
    print("Reading config file")

    # Create argument parser
    parser = argparse.ArgumentParser(description="Process configuration file for generating and storing embeddings.")

    # Add argument for configuration file
    parser.add_argument("config_file", type=str, help="Path to configuration file (e.g., config.yaml)")

    # Parse command-line arguments
    args = parser.parse_args()

    # Read configuration file
    config = reader.read_config(args.config_file)

    print("Config file data \n", yaml.dump(config, default_flow_style=False, sort_keys=False))

    generate_frames = config["generate_frames"]
    embed_frames = config["embed_frames"]
    path = config["videos"]  # args.videos_folder #
    image_output_dir = config["image_output_dir"]
    meta_output_dir = config["meta_output_dir"]
    N = config["number_of_frames_per_second"]
    chunk_duration = config["chunk_duration"]
    clip_duration = config["clip_duration"]

    host = VECTORDB_SERVICE_HOST_IP
    port = int(config["vector_db"]["port"])
    selected_db = config["vector_db"]["choice_of_db"]

    # Creating DB
    print(
        "Creating DB with text and image embedding support, \nIt may take few minutes to download and load all required models if you are running for first time."
    )
    print("Connect to {} at {}:{}".format(selected_db, host, port))

    vs = db.VS(host, port, selected_db)

    videos = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".mp4")]
    for video in videos:
        process_video(video, selected_db, chunk_duration, clip_duration)
