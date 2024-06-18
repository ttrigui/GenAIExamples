# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
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
from dateparser.search import search_dates

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     [%(asctime)s] %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S'
    )

# persistent flag
is_persistent = os.getenv("IS_PERSISTENT", "False")

@singleton
class DB_Handler:
    
    def __init__(self, configs):
        # initializing important variables
        self.configs = configs

        logging.info("Create text embedding model")
        self.text_embedder = SentenceTransformerEmbeddings(model_name=configs['text_embedder'])
        logging.info("Create image embedding model")
        self.image_embedder = OpenCLIPEmbeddings(model_name=configs['image_embedder'], checkpoint=configs['image_checkpoint'])
 
        # both chroma & vdms should be alive and stand by
        # hash map to store the client, db & retiever handshake.
        self.client = {}

        self.text_db = {"chroma":{},"vdms":{}}
        self.image_db = {"chroma":{},"vdms":{}}

        self.text_retriever = {"chroma":{},"vdms":{}}
        self.image_retriever = {"chroma":{},"vdms":{}}

        # initialize_db
        self.get_db_client()

        # for visual rag
        self.selected_db = "chroma"
        self.update_image_retriever = {"chroma": None,"vdms": None}

    def save_to_pkl_file(self, filename):
        """
        Save the db & retiever object to pickle file.
        """
        data_chroma = {
            "text_db": self.text_db["chroma"],
            "image_db": self.image_db["chroma"],
            "text_retriever": self.text_retriever["chroma"],
            "image_retriever": self.image_retriever["chroma"]
        }
        with open(filename, 'wb') as file:
            pickle.dump(data_chroma, file)

    def load_from_pkl_file(self, filename):
        """
        Load the database and retriever objects from a pickle file.
        Only for Chroma
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.text_db["chroma"] = data["text_db"]
            self.image_db["chroma"] = data["image_db"]
            self.text_retriever["chroma"] = data["text_retriever"]
            self.image_retriever["chroma"] = data["image_retriever"]
        
    def get_db_client(self):
        """
        Get the database clients.

        :return: None
        """
        vectordb_service_host_ip: str = os.getenv("VECTORDB_SERVICE_HOST_IP", "127.0.0.1")
        try:
            logging.info('Connecting to Chroma db server . . .')
            self.client["chroma"] = chromadb.HttpClient(
                settings=Settings(anonymized_telemetry=False),
                host=vectordb_service_host_ip,
                port=self.configs['chroma_service']['port']
            )
            logging.info('Connecting to VDMS db server . . .')
            self.client["vdms"] = VDMS_Client(
                host=vectordb_service_host_ip,
                port=self.configs['vdms_service']['port']
            )
        except Exception as e:
            logging.error(
                f"Client connection failed:\n"
                f"Chroma: {vectordb_service_host_ip}:{self.configs['chroma_service']['port']}\n"
                f"VDMS: {vectordb_service_host_ip}:{self.configs['vdms_service']['port']}\n"
                f"{e}",
                exc_info=True
            )

    def length(self, db_name: str, table: str, vtype: str):
        if db_name == 'chroma':
            if vtype == 'text':
                length = self.text_db["chroma"][table].__len__()
            elif vtype == 'image':
                length = self.image_db["chroma"][table].__len__()
            return length
        
        if db_name == 'vdms':
            pass
        
        return None
        
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
        if is_persistent:
            if db_name == 'chroma':
                logging.info("Saving to pickle file . . .")
                self.save_to_pkl_file(self.configs['handler_pickle_path'])
            else:
                pass
            
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
    
    def visual_rag_update_db(self, table, prompt, n_images):
        # for visual rag
        print ("Update DB")

        base_date = datetime.datetime.today()
        today_date= base_date.date()
        dates_found =search_dates(prompt, settings={"PREFER_DATES_FROM": "past", "RELATIVE_BASE": base_date})
        # if no date is detected dates_found should return None
        if dates_found != None:
            # Print the identified dates
            # print("dates_found:",dates_found)
            for date_tuple in dates_found:
                date_string, parsed_date = date_tuple
                print(f"Found date: {date_string} -> Parsed as: {parsed_date}")
                date_out = str(parsed_date.date())
                time_out = str(parsed_date.time())
                hours, minutes, seconds = map(float, time_out.split(":"))
                year, month, day_out = map(int, date_out.split("-"))
            
            # print("today's date", base_date)
            rounded_seconds = min(round(parsed_date.second + 0.5),59)
            parsed_date = parsed_date.replace(second=rounded_seconds, microsecond=0)

            # Convert the localized time to ISO format
            iso_date_time = parsed_date.isoformat()
            iso_date_time = str(iso_date_time)

            if self.selected_db == "vdms":
                if date_string == "today":
                    constraints = {"date": ["==", date_out]}
                    self.update_image_retriever["vdms"] = self.image_db["vdms"][table].as_retriever(
                        search_type="mmr", search_kwargs={"k": n_images, "filter": constraints}
                    )
                elif date_out != str(today_date) and time_out == "00:00:00":  ## exact day (example last friday)
                    constraints = {"date": ["==", date_out]}
                    self.update_image_retriever["vdms"] = self.image_db["vdms"][table].as_retriever(
                        search_type="mmr", search_kwargs={"k": n_images, "filter": constraints}
                    )
                elif (
                    date_out == str(today_date) and time_out == "00:00:00"
                ):  ## when search_date interprates words as dates output is todays date + time 00:00:00
                    self.update_image_retriever["vdms"] = self.image_db["vdms"][table].as_retriever(
                        search_type="mmr", search_kwargs={"k": n_images}
                    )
                else:  ## Interval  of time:last 48 hours, last 2 days,..
                    constraints = {"date_time": [">=", {"_date": iso_date_time}]}
                    self.update_image_retriever["vdms"] = self.image_db["vdms"][table].as_retriever(
                        search_type="mmr", search_kwargs={"k": n_images, "filter": constraints}
                    )
            if self.selected_db == "chroma":
                if date_string == "today":
                    self.update_image_retriever["chroma"] = self.image_db["chroma"][table].as_retriever(
                        search_type="mmr", search_kwargs={"k": n_images, "filter": {"date": {"$eq": date_out}}}
                    )
                elif date_out != str(today_date) and time_out == "00:00:00":  ## exact day (example last friday)
                    self.update_image_retriever["chroma"] = self.image_db["chroma"][table].as_retriever(
                        search_type="mmr", search_kwargs={"k": n_images, "filter": {"date": {"$eq": date_out}}}
                    )
                elif (
                    date_out == str(today_date) and time_out == "00:00:00"
                ):  ## when search_date interprates words as dates output is todays date + time 00:00:00
                    self.update_image_retriever["chroma"] = self.image_db["chroma"][table].as_retriever(
                        search_type="mmr", search_kwargs={"k": n_images}
                    )
                else:  ## Interval  of time:last 48 hours, last 2 days,..
                    constraints = {"date_time": [">=", {"_date": iso_date_time}]}
                    self.update_image_retriever["chroma"] = self.image_db["chroma"][table].as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "filter": {
                                "$or": [
                                    {
                                        "$and": [
                                            {"date": {"$eq": date_out}},
                                            {
                                                "$or": [
                                                    {"hours": {"$gte": hours}},
                                                    {
                                                        "$and": [
                                                            {"hours": {"$eq": hours}},
                                                            {"minutes": {"$gte": minutes}},
                                                        ]
                                                    },
                                                ]
                                            },
                                        ]
                                    },
                                    {
                                        "$or": [
                                            {"month": {"$gt": month}},
                                            {"$and": [{"day": {"$gt": day_out}}, {"month": {"$eq": month}}]},
                                        ]
                                    },
                                ]
                            },
                            "k": n_images,
                        },
                    )
        else:
            self.update_image_retriever[self.selected_db] = self.image_db[self.selected_db][table].as_retriever(search_type="mmr", search_kwargs={"k": n_images})

    def visual_rag_retrieval(
            self,
            table: str,
            prompt: str,
            n_images: Optional[int] = 3,
    ):
        '''
        visual RAG retrieval, use update_db for timestamp support
        '''
        self.visual_rag_update_db(table, prompt, n_images)
        image_results = self.update_image_retriever[self.selected_db].invoke(prompt)

        for r in image_results:
            print("images:", r.metadata["video"], "\t", r.metadata["date"], "\t", r.metadata["time"], "\n")

        return image_results
