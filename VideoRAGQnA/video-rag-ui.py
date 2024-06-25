import os

from embedding.vector_stores import db
import time
import torch

import torch
import streamlit as st
#from transformers import AutoTokenizer
from transformers import LlamaTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers import set_seed
import argparse

from typing import Any, List, Mapping, Optional
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import threading
from utils import config_reader as reader
from utils import prompt_handler as ph
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
from embedding.extract_vl_embedding import VLEmbeddingExtractor as VL
from embedding.generate_store_embeddings import setup_adaclip_model 
from embedding.video_llama.common.config import Config
from embedding.video_llama.common.dist_utils import get_rank
from embedding.video_llama.common.registry import registry
from embedding.video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

# imports modules for registration
from embedding.video_llama.datasets.builders import *
from embedding.video_llama.models import *
from embedding.video_llama.processors import *
from embedding.video_llama.runners import *
from embedding.video_llama.tasks import *

# Sets the random seed for reproducibility
set_seed(22)


# instructions = [
#     """Identify the person [with specific features / seen at a specific location / performing a specific action] in the provided data. 
#     Provide essential details such as their clothing, actions, and location. 
#     Exclude assumptions about age, specific items in the scene, or subjective observations. 
#     Ensure all information is distinct, accurate, and directly observable.

#     """,
    
#     """Identify the interactions between the man in a blue shirt and items in the provided data. 
#     Describe only the nature of his interactions with the items, such as picking up, reading, or moving them. 
#     Exclude assumptions about age, specific items in the scene, or subjective observations. 
#     Exclude any information about other individuals or unnecessary scene details. 
#     Ensure all information is distinct, accurate, and directly observable.
#     """,
    
#     """Provide a detailed, non-repetitive description of the actions and roles of individuals in the video. 
#     Focus on accurately describing each unique action and avoid mentioning the same detail multiple times. 
#     Ensure all information is directly observable and distinct.

#     """,
    
#     """Analyze the provided data to answer queries based on specific time intervals.
#     Provide detailed information corresponding to the specified time frames,
#     Do not give repetitions, always give distinct and accurate information only.""",
    
#     """When asked a question with a question mark, start the response with "yes" or "no" followed by a short, accurate explanation based on the video content. 
#     Describe only the relevant actions and appearances of individuals mentioned in the question. 
#     Ensure all information is directly observable, distinct, and exclude any unnecessary or incorrect details.
#     """,
    
#     """Answer questions related to events and activities that occurred on a specific day or hour of the day.
#     Provide a detailed account of the events,
#     Do not give repetitions, always give distinct and accurate information only."""
# ]

# instructions = [
#     """Identify the person [with specific features / seen at a specific location/ performing a specific action] in the provided data based on the visual content. 
#     Provide a detailed, non-repetitive description of their roles and actions. 
#     Focus on accurately describing each unique action and avoid mentioning the same detail multiple times.
#     Ensure all information is distinct, accurate, and directly observable. 
#     When asked a question with a question mark, start the response with "yes" or "no" followed by a short, accurate explanationwith respect to the question based on the video content.
#     Describe only the relevant actions and appearances of individuals mentioned in the question.
#     Exclude assumptions about age, specific items in the scene, or subjective observations.
#     """,
    
#     """Analyze the provided data to recognize and describe the activities performed by individuals.
#     Specify the type of activity and any relevant contextual details, 
#     Do not give repetitions, always give distinct and accurate information only.""",
    
#     """Determine the interactions between individuals and items in the provided data.
# #   Describe the nature of the interaction and the items involved. 
#     Exclude information about specific items on the shelf and scene, or subjective observations.
#     Do not give repetitions, always give distinct and accurate information only.
#     Ensure all information is directly observable, distinct, and exclude any unnecessary or incorrect details.
#     """,
    
#     """Analyze the provided data to answer queries based on specific time intervals.
#     Provide detailed information corresponding to the specified time frames,
#     Do not give repetitions, always give distinct and accurate information only.""",
    
#     """Identify individuals based on their appearance as described in the provided data.
#      Provide details about their identity and actions,
#      Do not give repetitions, always give distinct and accurate information only.""",
    
#     """Answer questions related to events and activities that occurred on a specific day.
#     Provide a detailed account of the events,
#     Do not give repetitions, always give distinct and accurate information only."""
# ]


instructions = [
    """Identify the person [with specific features / seen at a specific location
    / performing a specific action] in the provided data based on the visual content. 
    Provide details of their role, thier actions and their clothings.
    Describe only the relevant actions and appearances of individuals mentioned in the question.
    Ensure all information is distinct, accurate, and directly observable. 
    Provide a non-repetitive description of the actions performed by the person. 
    Exclude assumptions about age.
    Don not mention information about a variety of items and background information.
    When asked a question with a question mark, start the response with "yes" or "no" followed by a short, accurate explanation.
    Do not mention anything about a woman.
    """,
    
    """Analyze the provided data to recognize and describe the activities performed by individuals.
    Specify the type of activity and any relevant contextual details, 
    Do not give repetitions, always give distinct and accurate information only.""",
    
    """Determine the interactions between individuals and items in the provided data.
    Describe the nature of the interaction between individuals and items invloved.
    Exclude information about variety of items. Do not mention any item.
    Exclude assumptions about age, and all information about variety of items and background information.
    Do not mention anything about a woman.
    Do not give repetitions, always give distinct and accurate information only.""",
    
    """Analyze the provided data to answer queries based on specific time intervals.
    Provide detailed information corresponding to the specified time frames,
    Do not give repetitions, always give distinct and accurate information only.""",
    
    """Identify individuals based on their appearance as described in the provided data.
     Provide details about their identity and actions,
     Do not give repetitions, always give distinct and accurate information only.""",
    
    """Answer questions related to events and activities that occurred on a specific day.
    Provide a detailed account of the events,
    Do not give repetitions, always give distinct and accurate information only."""
]

# Embeddings - Initializes HuggingFace embedding
HFembeddings = HuggingFaceEmbeddings()

# Creates a FAISS vector store from the provided instructions using the embeddings.
hf_db = FAISS.from_texts(instructions, HFembeddings)

# Retrieves similar contexts from the vector store based on a query.
def get_context(query, hf_db=hf_db):
    context = hf_db.similarity_search(query)
    return [i.page_content for i in context]

if 'config' not in st.session_state.keys():
    st.session_state.config = reader.read_config('docs/config.yaml')

config = st.session_state.config
device = "cpu" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = config['model_path']
video_dir = config['videos']
# Read AdaCLIP
if not os.path.exists(os.path.join(config['meta_output_dir'], "metadata.json")):
    from embedding.generate_store_embeddings import main
    vs = main()
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

st.title("Video RAG")

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

@st.cache_resource       
def load_models():
    print("loading in model")
    #print("HF Token: ", HUGGINGFACEHUB_API_TOKEN)
    #model = AutoModelForCausalLM.from_pretrained(
    #    model_path, torch_dtype=torch.float32, device_map=device, trust_remote_code=True, token=HUGGINGFACEHUB_API_TOKEN
    #)

    #tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=HUGGINGFACEHUB_API_TOKEN)
    #tokenizer.padding_size = 'right'

    # Load video-llama model
    video_llama = VL(**config['vl_branch'])
    tokenizer = video_llama.model.llama_tokenizer
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    
    return video_llama, tokenizer, streamer

video_llama, tokenizer, streamer = load_models()
vis_processor_cfg = video_llama.cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
print("-"*30)
print("initializing model")
chat = Chat(video_llama.model, vis_processor, device=device)

def chat_reset(chat_state, img_list):
    print("-"*30)
    print("resetting chatState")
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

#img_list = []
#chat_state = conv_llava_llama_2.copy()
#chat.upload_video_without_audio(video_path, chat_state, img_list)
#_, chat_state = chat.ask(text_input, chat_state)
#answer, chat_state, img_list = chat.answer(chat_state, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=0.1, max_length=2000, keep_conv_hist=True)
#chat_state, img_list = chat_reset(chat_state, img_list)

class VideoLLM(LLM):
        
    @torch.inference_mode()
    def _call(
            self, 
            video_path,
            text_input,
            chat,
            start_time,
            duration,
            streamer = None,  # Add streamer as an argument
        ):
        
        print(" - - ")
        print("  text_input:", text_input)
        print(" - - ")
        
        chat.upload_video_without_audio(video_path, start_time, duration)
        chat.ask(text_input)#, chat_state)
        #answer = chat.answer(chat_state, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=0.1, max_length=2000, keep_conv_hist=True, streamer=streamer)
        answer = chat.answer(max_new_tokens=150, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=0.02, max_length=2000, keep_conv_hist=True, streamer=streamer)

    def stream_res(self, video_path, text_input, chat, start_time, duration):
        #thread = threading.Thread(target=self._call, args=(video_path, text_input, chat, chat_state, img_list, streamer))  # Pass streamer to _call
        thread = threading.Thread(target=self._call, args=(video_path, "<rag_prompt>"+text_input, chat, start_time, duration, streamer))  # Pass streamer to _call
        thread.start()
        
        for text in streamer:
            yield text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return model_path # {"name_of_model": model_path}

    @property
    def _llm_type(self) -> str:
        return "custom"

class CustomLLM(LLM):
        
    @torch.inference_mode()
    def _call(
            self, 
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            streamer: Optional[TextIteratorStreamer] = None,  # Add streamer as an argument
        ) -> str:
        
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        print(" - - ")
        print("  prompt:", prompt)
        print(" - - ")
        
        with torch.no_grad():
            print(model.device)
            print(tokens.device)
            output = model.generate(input_ids = tokens,
                                    max_new_tokens = 100,
                                    num_return_sequences = 1,
                                    num_beams = 1,
                                    min_length = 1,
                                    top_p = 0.9,
                                    top_k = 50,
                                    repetition_penalty = 1.2,
                                    length_penalty = 1,
                                    temperature = 0.1,
                                    streamer=streamer,
                                    # pad_token_id=tokenizer.eos_token_id,
                                    do_sample=True
                    )
        
    def stream_res(self, prompt):
        thread = threading.Thread(target=self._call, args=(prompt, None, None, streamer))  # Pass streamer to _call
        thread.start()
        
        for text in streamer:
            yield text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return model_path # {"name_of_model": model_path}

    @property
    def _llm_type(self) -> str:
        return "custom"
    
def get_top_doc(results, qcnt):
    print("-"*30)
    print("retrieving videos model")
    hit_score = {}
    for r in results:
        try:
            video_name = r.metadata['video']
            #playback_offset = r.metadata["start of interval in sec"]
            playback_offset = r.metadata["timestamp"]
            if video_name not in hit_score.keys(): hit_score[video_name] = 0
            hit_score[video_name] += 1
        except:
            r,score = r
            video_name = r.metadata['video_path']
            #playback_offset = r.metadata["start of interval in sec"]
            playback_offset = r.metadata["timestamp"]
            hit_score[video_name] = score

    x = dict(sorted(hit_score.items(), key=lambda item: -item[1]))
    
    if qcnt >= len(x):
        return None, None
    print (f'top docs = {x}')
    print("-"*30)
    print("video retrieval done")
    return {'video': list(x)[qcnt]}, playback_offset

def play_video(x, offset):
    if x is not None:
        #video_file = x.replace('.pt', '')
        #path = video_dir + video_file
        #video_file = open(path, 'rb')
        video_file = open(x, 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes, start_time=offset)

if 'llm' not in st.session_state.keys():
    with st.spinner('Loading Models . . .'):
        time.sleep(1)
        if config['embeddings']['type'] == "frame":
            print("Loading CustomLLM . . .")
            st.session_state['llm'] = CustomLLM()
        elif config['embeddings']['type'] == "video":
            st.session_state['llm'] = VideoLLM()
            print("Loading VideoLLM . . .")
        else:
            print("ERROR: line 240")
        
if 'vs' not in st.session_state.keys():
    with st.spinner('Preparing RAG pipeline'):
        time.sleep(1)
        host = st.session_state.config['vector_db']['host']
        port = int(st.session_state.config['vector_db']['port'])
        selected_db = st.session_state.config['vector_db']['choice_of_db']
        try:
            st.session_state['vs'] = vs
        except:
            if config['embeddings']['type'] == "frame":
                st.session_state['vs'] = db.VS(host, port, selected_db)
            elif config['embeddings']['type'] == "video":
                import json
                adaclip_cfg_json = json.load(open(config['adaclip_cfg_path'], 'r'))
                adaclip_cfg_json["resume"] = config['adaclip_model_path']
                adaclip_cfg = argparse.Namespace(**adaclip_cfg_json)
                model, _ = setup_adaclip_model(adaclip_cfg, device=device)
                st.session_state['vs'] = db.VideoVS(host, port, selected_db, model) # FIX THIS LINE

        if st.session_state.vs.client == None:
            print ('Error while connecting to vector DBs')
        
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        
def clear_chat_history():
    st.session_state.example_video = 'Enter Text'
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    chat.clear()
        
def RAG(prompt):
    
    with st.status("Querying database . . . ", expanded=True) as status:
        st.write('Retrieving 3 image docs') #1 text doc and 
        results = st.session_state.vs.MultiModalRetrieval(prompt, top_k = 3) #n_texts = 1, n_images = 3)
        status.update(label="Retrived Top matching video!", state="complete", expanded=False)
    print("---___---")
    print (f'\tRAG prompt={prompt}')
    print("---___---")
      
    top_doc, playback_offset = get_top_doc(results, st.session_state["qcnt"])
    print ('TOP DOC = ', top_doc)
    print("PLAYBACK OFFSET = ", playback_offset)
    if top_doc == None:
        return None, None, None
    video_name = top_doc['video']
    print('Video from top doc: ', video_name)
    
    return video_name, playback_offset, top_doc

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
    print("-"*30)
    print("starting message handling")
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
            video_name, playback_offset, top_doc = RAG(prompt)
            print("VIDEO NAME USED IN PLAYBACK: ", video_name)
            if video_name == None:
                full_response = f"No more relevant videos found. Select a different query. \n\n"
                placeholder.markdown(full_response)
                end = time.time()
            else:
                with col2:
                    play_video(video_name, playback_offset)
                """
                scene_des = get_description(video_name)
                formatted_prompt = ph.get_formatted_prompt(scene=scene_des, prompt=prompt)
                """
                
                full_response = ''
                full_response = f"Most relevant retrived video is **{video_name}** \n\n"
                instruction = f"{get_context(prompt)[0]}: {prompt}"
                #for new_text in st.session_state.llm.stream_res(formatted_prompt):
                for new_text in st.session_state.llm.stream_res(video_name, instruction, chat, playback_offset, config['clip_duration']):
                    full_response += new_text
                    placeholder.markdown(full_response)

                end = time.time()
                full_response += f'\n\n🚀 Generated in {(end - start):.4f} seconds.'
                #chat_state, img_list = chat_reset(chat_state, img_list)
                #chat.clear()
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        print("-"*30)
        print("message handling done")
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
            'Man holding red shopping basket',
            'Was there any person wearing a blue shirt seen today?',
            'Was there any person wearing a blue shirt seen in the last 6 hours?',
            #'Was there any person wearing a blue shirt seen last Sunday?',
            #'Was a person wearing glasses seen in the last 30 minutes?',
            'Was a person wearing glasses seen in the last 72 hours?',
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
