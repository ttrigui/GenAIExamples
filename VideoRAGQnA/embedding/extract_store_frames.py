import os
import time as t
from tqdm import tqdm
import cv2
import json
import datetime
import random
from tzlocal import get_localzone

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
    
def random_date_in_past_6_months():
    # Get the current date and time
    now = datetime.datetime.now()
    
    # Calculate the date 6 months ago from now
    six_months_ago = now - datetime.timedelta(days=6*30)
    
    # Generate a random date between six_months_ago and now
    random_date = six_months_ago + (now - six_months_ago) * random.random()
    
    return random_date
    
def process_all_videos(config):
    path = config['videos']
    meta_output_dir = config['meta_output_dir']
    selected_db = config['vector_db']['choice_of_db']
    emb_path = config['embeddings']['path']
    emb_type = config['embeddings']['type']
    chunk_duration = config['chunk_duration']
    clip_duration = config['clip_duration']

    videos = [file for file in os.listdir(path) if file.endswith('.mp4')] # TODO: increase supported video formats

    # print (f'Total {len(videos)} videos will be processed')
    metadata = {}
    
    for i, each_video in enumerate(tqdm(videos)):
        metadata[each_video] = {}
        keyname = each_video
        video_path = os.path.join(path, each_video)
        # for current datetime use datetime.datetime.now() 
        date_time = random_date_in_past_6_months()
        # Get the local timezone of the machine
        local_timezone = get_localzone()
        if emb_type == 'video':
            time_format = "%a %b %d %H:%M:%S %Y"
            if not isinstance(date_time, datetime.datetime):
                date_time = datetime.datetime.strptime(date_time, time_format)
            time = date_time.strftime("%H:%M:%S")
            hours, minutes, seconds = map(float, time.split(":"))
            date = date_time.strftime("%Y-%m-%d")
            year, month, day = map(int, date.split("-"))

            if clip_duration is not None and chunk_duration is not None and clip_duration <= chunk_duration:
                interval_count = 0
                metadata.pop(each_video)
                for start_frame, end_frame, start_time, end_time in calculate_intervals(video_path, chunk_duration, clip_duration):
                    keyname = os.path.splitext(os.path.basename(video_path))[0]+f"_interval_{interval_count}"
                    metadata[keyname] = {"timestamp":start_time}
                    metadata[keyname].update({"date": date, "year": year, "month": month, "day": day, 
                        "time": time, "hours": hours, "minutes": minutes, "seconds": seconds})
                    if selected_db == 'vdms':
                        # Localize the current time to the local timezone of the machine
                        #Tahani might not need this
                        current_time_local = date_time.replace(tzinfo=datetime.timezone.utc).astimezone(local_timezone)

                        # Convert the localized time to ISO 8601 format with timezone offset
                        iso_date_time = current_time_local.isoformat()
                        metadata[keyname]['date_time'] = {"_date": str(iso_date_time)}

                    # Open the video file
                    cap = cv2.VideoCapture(video_path)

                    if int(cv2.__version__.split('.')[0]) < 3:
                        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # get the duration
                    metadata[keyname].update({
                            "clip_duration":(min(total_frames,end_frame)-start_frame)/fps,
                            'fps': fps, 
                            'total_frames': total_frames, 
                            #'embedding_path': os.path.join(emb_path, each_video+".pt"),
                            'video_path': f'{os.path.join(path,each_video)}',
                        })
                    cap.release()
                    interval_count+=1
        metadata[keyname].update({
                'fps': fps, 
                'total_frames': total_frames, 
                #'embedding_path': os.path.join(emb_path, each_video+".pt"),
                'video_path': f'{os.path.join(path,each_video)}',
            })
    os.makedirs(meta_output_dir, exist_ok=True)
    metadata_file = os.path.join(meta_output_dir, f"metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)