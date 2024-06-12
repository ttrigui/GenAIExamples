# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time
from typing import Any, List

import streamlit as st
from langchain.llms.base import LLM
from transformers import set_seed
from utils import config_reader as reader
from utils import prompt_handler as ph
import requests
from queue import Queue, Empty
import logging

def set_proxy(addr:str):
    # for DNS: "http://child-prc.intel.com:913"
    # for Huggingface downloading: "http://proxy-igk.intel.com:912"
    os.environ['http_proxy'] = addr
    os.environ['https_proxy'] = addr
    os.environ['HTTP_PROXY'] = addr
    os.environ['HTTPS_PROXY'] = addr

def get_data(api_url:str, query:dict):
    try:
        set_proxy("http://child-prc.intel.com:913")
        response = requests.get(api_url, query)
        response.raise_for_status()
        set_proxy("http://proxy-igk.intel.com:912")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        return None

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     [%(asctime)s] %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S"
    )

# from vector_stores import db
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

set_seed(22)

set_proxy("http://proxy-igk.intel.com:912")

if "config" not in st.session_state.keys():
    st.session_state.config = reader.read_config("./config.yaml")

config = st.session_state.config

model_path = config["model_path"]
video_dir = config["videos"]
video_dir = video_dir.replace("../", "")
model_server_url = "http://" + config['model_server']['host'] + ':' + str(config['model_server']['port']) + config['model_server']['url'] 
vector_query_url = "http://" + config['vector_db']['host'] + ':' + str(config['vector_db']['port']) + config['vector_db']['vector_query_url'] 

st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

st.title("Video RAG")

title_alignment = """
<style>
h1 {
  text-align: center
}

video.stVideo {
    width: 200px;
    height: 500px;
}
</style>
"""
st.markdown(title_alignment, unsafe_allow_html=True)

class ModelServerLLM(LLM):

    def _call(self, prompt: str, q: Queue):
        set_proxy("http://child-prc.intel.com:913")
        request_body = {"text": prompt, "config": {}, "stream": True}
        
        response = requests.post(
            model_server_url,
            json=request_body,
            stream=True
        )
        if response.status_code == 200:
            for text in response.iter_content(chunk_size=None, decode_unicode=True):
                if text:
                    q.put(text)
        else:
            raise ValueError(f"Failed to get response from model server: {response.status_code}")
        set_proxy("http://proxy-igk.intel.com:912")

    def stream_res(self, prompt):
        q = Queue()
        thread = threading.Thread(target=self._call, args=(prompt, q))
        thread.start()
        
        while True:
            try:
                chunk = q.get(timeout=1)
                if chunk is None:  # End of stream
                    break
                yield chunk
            except Empty:
                if not thread.is_alive():
                    break

    @property
    def _identifying_params(self):
        return {"model_server_url": model_server_url}

    @property
    def _llm_type(self):
        return "model_server"

def get_top_doc(results, qcnt):
    hit_score = {}
    for r in results:
        try:
            video_name = r["metadata"]["video"]
            if video_name not in hit_score.keys():
                hit_score[video_name] = 0
            hit_score[video_name] += 1
        except:
            pass

    x = dict(sorted(hit_score.items(), key=lambda item: -item[1]))

    if qcnt >= len(x):
        return None
    print(f"top docs = {x}")
    return {"video": list(x)[qcnt]}


def play_video(x):
    if x is not None:
        video_file = x.replace(".pt", "")
        path = video_dir + video_file

        video_file = open(path, "rb")
        video_bytes = video_file.read()

        st.video(video_bytes, start_time=0)


if "llm" not in st.session_state.keys():
    with st.spinner("Loading Models . . ."):
        time.sleep(1)
        st.session_state["llm"] = ModelServerLLM()

# if "vs" not in st.session_state.keys():
#     with st.spinner("Preparing RAG pipeline"):
#         time.sleep(1)
#         host = st.session_state.config["vector_db"]["host"]
#         port = int(st.session_state.config["vector_db"]["port"])
#         selected_db = st.session_state.config["vector_db"]["choice_of_db"]
#         st.session_state["vs"] = db.VS(host, port, selected_db)

#         if st.session_state.vs.client == None:
#             print("Error while connecting to vector DBs")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


def clear_chat_history():
    st.session_state.example_video = "Enter Text"
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


def RAG(prompt):

    with st.status("Querying database . . . ", expanded=True) as status:
        st.write("Retrieving 3 image docs")  # 1 text doc and
        # results = st.session_state.vs.MultiModalRetrieval(prompt, n_images=3)  # n_texts = 1, n_images = 3)
        results = get_data(vector_query_url, {"prompt": prompt})
        status.update(label="Retrieved Top matching video!", state="complete", expanded=False)

    logging.info (f"prompt = {prompt}")

    top_doc = get_top_doc(results, st.session_state["qcnt"])

    logging.info (f"TOP DOC = {top_doc}")
    if top_doc == None:
        return None, None
    video_name = top_doc["video"]

    return video_name, top_doc


def get_description(vn):
    content = None
    des_path = os.path.join(config["description"], vn + ".txt")
    with open(des_path, "r") as file:
        content = file.read()
    return content


def get_history():
    messages = st.session_state.messages
    return messages[-3:]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

if "prevprompt" not in st.session_state.keys():
    st.session_state["prevprompt"] = ""
    print("Setting prevprompt to None")
if "prompt" not in st.session_state.keys():
    st.session_state["prompt"] = ""
if "qcnt" not in st.session_state.keys():
    st.session_state["qcnt"] = 0


def handle_message():
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        # Handle user messages here
        with st.chat_message("assistant"):
            placeholder = st.empty()
            start = time.time()
            prompt = st.session_state["prompt"]

            if prompt == "Find similar videos":
                prompt = st.session_state["prevprompt"]
                st.session_state["qcnt"] += 1
            else:
                st.session_state["qcnt"] = 0
                st.session_state["prevprompt"] = prompt
            video_name, top_doc = RAG(prompt)
            if video_name == None:
                full_response = "No more relevant videos found. Select a different query. \n\n"
                placeholder.markdown(full_response)
                end = time.time()
            else:
                with col2:
                    play_video(video_name)

                scene_des = get_description(video_name)
                logging.info (f"scene_des = {scene_des}")

                formatted_prompt = ph.get_formatted_prompt(scene=scene_des, prompt=prompt, history=get_history())

                full_response = ""
                full_response = f"Most relevant retrieved video is **{video_name}** \n\n"

                logging.info (f"formatted_prompt = {formatted_prompt}")
                for new_text in st.session_state.llm.stream_res(formatted_prompt):
                    full_response += new_text
                    placeholder.markdown(full_response)

                end = time.time()
                full_response += f"\n\n🚀 Generated in {end - start} seconds."
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)


def display_messages():
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


col1, col2 = st.columns([2, 1])

with col1:
    st.selectbox(
        "Example Prompts",
        (
            "Enter Text",
            "Find similar videos",
            "Man wearing glasses",
            "People reading item description",
            "Man holding red shopping basket",
            "Was there any person wearing a blue shirt seen today?",
            "Was there any person wearing a blue shirt seen in the last 6 hours?",
            "Was there any person wearing a blue shirt seen last Sunday?",
            "Was a person wearing glasses seen in the last 30 minutes?",
            "Was a person wearing glasses seen in the last 72 hours?",
        ),
        key="example_video",
    )

    st.write("You selected:", st.session_state.example_video)


if st.session_state.example_video == "Enter Text":
    if prompt := st.chat_input(disabled=False):
        st.session_state["prompt"] = prompt
        # st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        if prompt == "Find similar videos":
            st.session_state.messages.append({"role": "assistant", "content": "Not supported"})
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
else:
    prompt = st.session_state.example_video
    st.session_state["prompt"] = prompt
    # st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.chat_input(disabled=True)
    if prompt == "Find similar videos":
        st.session_state.messages.append({"role": "user", "content": prompt + ": " + st.session_state["prevprompt"]})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

with col1:
    display_messages()
    handle_message()