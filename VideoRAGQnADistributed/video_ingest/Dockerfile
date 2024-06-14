ARG OS_VER=22.04
ARG OS_IMAGE=ubuntu

FROM ${OS_IMAGE}:${OS_VER}
USER root
WORKDIR /video_ingest

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    python3.9 python3.9-venv libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# avoid installing packages while re-build
COPY ./video_ingest/requirements.txt requirements.txt

RUN python3.9 -m venv /root/myenv

RUN . /root/myenv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    deactivate

COPY ./video_ingest/scripts scripts/
COPY ./utils scripts/utils/
COPY ./conf scripts/

COPY ./video_ingest/videos videos/
COPY ./video_ingest/scene_description scene_description/

ENTRYPOINT ["/root/myenv/bin/python3.9","/video_ingest/scripts/generate_store_embeddings.py", "/video_ingest/scripts/config.yaml", "/video_ingest/videos/"]