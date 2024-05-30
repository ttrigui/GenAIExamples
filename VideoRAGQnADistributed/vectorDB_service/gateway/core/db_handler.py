import os
from typing import List, Optional, Iterable
import pickle
from singleton_decorator import singleton
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma, VDMS
from langchain_community.vectorstores.vdms import VDMS_Client
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.runnables import ConfigurableField

@singleton
class DB_Handler:
    
    def __init__(self, configs):
        # initializing important variables
        self.configs = configs

        # TODO: set proxy here if download issue
        self.text_embedder = SentenceTransformerEmbeddings(model_name=configs['text_embedder'])
        self.image_embedder = OpenCLIPEmbeddings(model_name=configs['image_embedder'], checkpoint=configs['image_checkpoint'])
        print("done image embedding init")
        self.set_proxy("")
 
        # both chroma & vdms should be alive and stand by
        # hash map to store the client, db & retiever handshake.
        self.client = {}

        self.text_db = {"chroma":{},"vdms":{}}
        self.image_db = {"chroma":{},"vdms":{}}

        self.text_retriever = {"chroma":{},"vdms":{}}
        self.image_retriever = {"chroma":{},"vdms":{}}

        # initialize_db
        self.get_db_client()

        # video-llama
        self.selected_db = "chroma"

    def set_proxy(self, addr:str):
        os.environ['http_proxy'] = addr
        os.environ['https_proxy'] = addr
        os.environ['HTTP_PROXY'] = addr
        os.environ['HTTPS_PROXY'] = addr

    def save_to_pkl_file(self, filename):
        """
        Save the db & retiever object to pickle file.
        """
        data = {
            "text_db": self.text_db,
            "image_db": self.image_db,
            "text_retriever": self.text_retriever,
            "image_retriever": self.image_retriever
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def load_from_pkl_file(self, filename):
        """
        Load the database and retriever objects from a pickle file.
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.text_db = data["text_db"]
            self.image_db = data["image_db"]
            self.text_retriever = data["text_retriever"]
            self.image_retriever = data["image_retriever"]
        
    def get_db_client(self):
        # prepare all client
        print ('Connecting to Chroma db server . . .')
        self.client["chroma"] = chromadb.HttpClient(settings=Settings(anonymized_telemetry=False), 
                                                    host=self.configs['chroma_service']['host'], 
                                                    port=self.configs['chroma_service']['port'])
        print ('Connecting to VDMS db server . . .')
        self.client["vdms"] = VDMS_Client(host=self.configs['vdms_service']['host'], port=self.configs['vdms_service']['port'])

    def length(self, db_name: str, table: str):
        if db_name == 'chroma':
            texts_len = self.text_db["chroma"][table].__len__()
            images_len = self.image_db["chroma"][table].__len__()
            return (texts_len, images_len)
        
        if db_name == 'vdms':
            pass
        
        return (None, None)
        
    def add_table(self, db_name: str, table: str, vtype: str):
        if vtype == 'text':
            if db_name ==  'chroma':
                self.text_db[db_name][table] = Chroma(
                    client = self.client[db_name],
                    embedding_function = self.text_embedder,
                    collection_name = table,
                    persist_directory ="/chroma/chroma",
                )
            if db_name == 'vdms':
                self.text_db[db_name][table] = VDMS (
                    client = self.client[db_name],
                    embedding = self.text_embedder,
                    collection_name = table,
                    engine = "FaissFlat",
                )
                
            self.text_retriever[db_name][table] = self.text_db[db_name][table].as_retriever().configurable_fields(
                search_kwargs=ConfigurableField(
                    id="k_text_docs",
                    name="Search Kwargs",
                    description="The search kwargs to use",
                )
            )
        elif vtype == 'image':
            if db_name ==  'chroma':
                self.image_db[db_name][table] = Chroma(
                    client = self.client[db_name],
                    embedding_function = self.image_embedder,
                    collection_name = table,
                    persist_directory ="/chroma/chroma",
                )

            if db_name == 'vdms':
                self.image_db[db_name][table] = VDMS (
                    client = self.client[db_name],
                    embedding = self.image_embedder,
                    collection_name = table,
                    engine = "FaissFlat",
                )
                
            self.image_retriever[db_name][table] = self.image_db[db_name][table].as_retriever(search_type="mmr").configurable_fields(
                search_kwargs=ConfigurableField(
                    id="k_image_docs",
                    name="Search Kwargs",
                    description="The search kwargs to use",
                )
            )
        self.save_to_pkl_file(self.configs['handler_pickle_path'])
            
    def delete_collection(self, db_name:str, table:str):
        self.client[db_name].delete_collection(table)
      
    def add_images(
            self,
            db_name: str,
            table: str,
            uris: List[str],
            metadatas: Optional[List[dict]] = None,
        ):

        self.image_db[db_name][table].add_images(uris, metadatas)

    def add_texts(
            self,
            db_name: str,
            table: str,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
        ):

        self.text_db[db_name][table].add_texts(texts, metadatas)

    def multi_modal_retrieval(
            self,
            db_name: str,
            text_table: str,
            image_table: str,
            query: str,
            n_texts: Optional[int] = 1,
            n_images: Optional[int] = 3,
        ):
        
        text_config = {"configurable": {"k_text_docs": {"k": n_texts}}}
        image_config = {"configurable": {"k_image_docs": {"k": n_images}}}

        text_results = self.text_retriever[db_name][text_table].invoke(query, config=text_config)
        image_results = self.image_retriever[db_name][image_table].invoke(query, config=image_config)

        return text_results + image_results
    
    def single_modal_retrieval(
            self,
            db_name: str,
            table: str,
            vtype: str,
            query: str,
            n: Optional[int] = 1,
        ):
        
        user_config = {"configurable": {f"k_{vtype}_docs": {"k": n}}}
        if vtype == 'text':
            results = self.text_retriever[db_name][table].invoke(query, config=user_config)
        elif vtype == 'image':
            results = self.image_retriever[db_name][table].invoke(query, config=user_config)
        
        return results
