import requests
import os

def set_proxy(addr:str):
    # for DNS: "http://child-prc.intel.com:913"
    # for Huggingface downloading: "http://proxy-igk.intel.com:912"
    os.environ['http_proxy'] = addr
    os.environ['https_proxy'] = addr
    os.environ['HTTP_PROXY'] = addr
    os.environ['HTTPS_PROXY'] = addr


def post_data(api_url: str, body:dict):
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(api_url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        return None


def get_data(api_url:str, params:dict):
    try:
        # headers = {'Content-Type': 'application/json'}
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raises HTTPError for bad responses
        return response.json()  # Returns the json content of the response
    except requests.exceptions.RequestException as e:
        print(e)
        return None

def init_vectordb(db_kind:str):
    api_url = "http://172.16.186.168:9001/video_llama_retriever/init_db"
    results = post_data(api_url, {"selected_db": db_kind})
    return results

set_proxy("http://child-prc.intel.com:913")
# get, works
# results = get_data("http://172.16.186.168:9001/video_llama_retriever/query", {'prompt': 'Man wearing glasses"'})
# api_url = "http://172.16.186.168:9001/video_llama_retriever/add_images"
# results = post_data(api_url, {"uris": ["11","22"], "metadatas": []})

# api_url = "http://172.16.186.168:9001/video_llama_retriever/add_texts"
# results = post_data(api_url, {"texts": ["11","22"], "metadatas": []})
# results = init_vectordb("chroma")
# set_proxy("http://proxy-igk.intel.com:912")


# 建表
# api_url = 'http://172.16.186.168:9001/add_table'
# results = post_data(api_url, {"db_name": "chroma","table": "hhjiji","type": "text"})
# print(results)


# 插字
# api_url = 'http://172.16.186.168:9001/add_texts'
# results = post_data(api_url, {"db_name": "chroma","table": "hhjiji","texts":["变态", "大便", "智障", "傻逼"]})


# 查询
api_url = 'http://172.16.186.168:9001/search/'
results = get_data(api_url, {"db_name": "chroma","table": "hhjiji","type": "text","query":"你喝不喝尿阿 ","n_results": 1})

print(results)