import os
import time
import streamlit as st
from typing import Any, List
from langchain.llms.base import LLM
import threading
from transformers import set_seed
from utils import config_reader as reader
from utils import prompt_handler as ph
import requests
from queue import Queue, Empty

set_seed(22)

if 'config' not in st.session_state.keys():
    st.session_state.config = reader.read_config('./docs/config.yaml')

config = st.session_state.config

model_path = config['model_path']
video_dir = config['videos']

st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

st.title("Visual RAG")

title_alignment="""
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
        request_body = {"text": prompt, "config": {}, "stream": True}
        
        response = requests.post(
            config["model_server_url"],
            json=request_body,
            stream=True
        )
        if response.status_code == 200:
            for text in response.iter_content(chunk_size=None, decode_unicode=True):
                if text:
                    q.put(text)
        else:
            raise ValueError(f"Failed to get response from model server: {response.status_code}")
    

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
        return {"model_server_url": self.model_server_url}

    @property
    def _llm_type(self):
        return "custom"


def get_data(api_url:str, query:dict):
    try:
        response = requests.get(api_url, query)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        return None

def get_top_doc(results:dict, qcnt):
    hit_score = {}
    for r in results:
        try:
            video_name = r['metadata']['video']
            if video_name not in hit_score.keys(): hit_score[video_name] = 0
            hit_score[video_name] += 1
        except:
            pass

    x = dict(sorted(hit_score.items(), key=lambda item: -item[1]))
    if qcnt >= len(x):
        return None
    print (f'top docs = {x}')
    return {'video': list(x)[qcnt]}

def play_video(x):
    if x is not None:
        video_file = x.replace('.pt', '')
        path = video_dir + video_file
        
        video_file = open(path, 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes, start_time=0)

if 'llm' not in st.session_state.keys():
    with st.spinner('Loading Models . . .'):
        time.sleep(1)
        # TODO: change LLM / model server here
        st.session_state['llm'] = ModelServerLLM()
        
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        
def clear_chat_history():
    st.session_state.example_video = 'Enter Text'
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        
def RAG(prompt):
    """
    give query, return top hit video name string
    """
    with st.status("Querying database . . . ", expanded=True) as status:
        st.write('Retrieving 1 text doc and 3 image docs')

        results = get_data(config['vector_query_url'], {"prompt": prompt})
        status.update(label="Retrived Top matching video!", state="complete", expanded=False)

    print (f'promt={prompt}\n')

    top_doc = get_top_doc(results, st.session_state['qcnt'])
    print ('TOP DOC = ', top_doc)
    if top_doc == None:
        return None, None
    video_name = top_doc['video']
    print("video_name", video_name)
    print("top_doc", top_doc)
    return video_name, top_doc

def get_description(vn):
    content = None
    des_path = os.path.join(config['description'], vn + '.txt')
    with open(des_path, 'r') as file:
        content = file.read()
    return content
    
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if 'prevprompt' not in st.session_state.keys():
    st.session_state['prevprompt'] = ''
    print("Setting prevprompt to None")
if 'prompt' not in st.session_state.keys():
    st.session_state['prompt'] = ''
if 'qcnt' not in st.session_state.keys():
    st.session_state['qcnt'] = 0

def handle_message():
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        # Handle user messages here
        with st.chat_message("assistant"):
            placeholder = st.empty()
            start = time.time()
            prompt = st.session_state['prompt']
            
            if prompt == 'Find similar videos':
                prompt = st.session_state['prevprompt']
                st.session_state['qcnt'] += 1
            else:
                st.session_state['qcnt'] = 0
                st.session_state['prevprompt'] = prompt

            video_name, top_doc = RAG(prompt)

            if video_name == None:
                full_response = f"No more relevant videos found. Select a different query. \n\n"
                placeholder.markdown(full_response)
                end = time.time()
            else:
                with col2:
                    play_video(video_name)
                
                scene_des = get_description(video_name)
                print("-"*100)
                print("scence description:",scene_des)
                print("-"*100)
                formatted_prompt = ph.get_formatted_prompt(scene=scene_des, prompt=prompt)
                
                full_response = ''
                full_response = f"Microservice version: Most relevant retrived video is **{video_name}** \n\n"

                # TODO: LLM response with prompt
                for new_text in st.session_state.llm.stream_res(formatted_prompt):
                    full_response += new_text
                    placeholder.markdown(full_response)
                
                end = time.time()
                full_response += f'\n\n🚀 Generated in {end - start} seconds.'
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
        'Example Prompts',
        (
            'Enter Text', 
            'Find similar videos', 
            'Man wearing glasses', 
            'People reading item description',
            'Man wearing khaki pants',
            'Man laughing',
            'Black tshirt guy holding red basket',
            'Man holding red shopping basket',
            'Man wearing blue shirt',
            'Man putting object into his pocket',
        ),
        key='example_video'
    )

    st.write('You selected:', st.session_state.example_video)

if st.session_state.example_video == 'Enter Text':
    if prompt := st.chat_input(disabled=False):
        st.session_state['prompt'] = prompt
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        if prompt == 'Find similar videos':            
            st.session_state.messages.append({"role": "assistant", "content": "Not supported"})
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
        #st.session_state.messages.append({"role": "user", "content": prompt})
else:
    prompt = st.session_state.example_video
    st.session_state['prompt'] = prompt
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.chat_input(disabled=True)
    if prompt == 'Find similar videos':
        st.session_state.messages.append({"role": "user", "content": prompt+': '+st.session_state['prevprompt']})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

with col1:
    display_messages()
    handle_message()
