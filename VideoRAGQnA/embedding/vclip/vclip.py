import yaml
import json
import os, sys
import argparse
import torch
import numpy as np
from decord import VideoReader, cpu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, AutoProcessor, CLIPModel
import torchvision.transforms as T
toPIL = T.ToPILImage()

import torch.nn as nn
from einops import rearrange

class vCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_frm = cfg["num_frm"]
        self.model_name = cfg["model_name"]

        self.clip = CLIPModel.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


    def get_text_embeddings(self, texts):
        """
        input is list of texts
        """
        text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        text_features = self.clip.get_text_features(**text_inputs)
        return text_features


    def get_image_embeddings(self, images):
        """
        input is list of images
        """
        image_inputs = self.processor(images=images, return_tensors="pt")
        image_features = self.clip.get_image_features(**image_inputs)
        return image_features 


    def get_video_embeddings(self, frames_batch):
        """
        input is list of list of frames in video
        """
        self.batch_size = len(frames_batch)
        vid_embs = []
        for frames in frames_batch:
            frame_embeddings = self.get_image_embeddings(frames)
            frame_embeddings = rearrange(frame_embeddings, "(b n) d -> b n d", b=len(frames_batch))
            # Normalize, mean aggregate and return normalized video_embeddings
            frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True) 
            video_embeddings = frame_embeddings.mean(dim=1)
            video_embeddings = video_embeddings / video_embeddings.norm(dim=-1, keepdim=True)
            vid_embs.append(video_embeddings)
        return torch.cat(vid_embs, dim=0)

