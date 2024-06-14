ARG OS_VER=22.04
ARG OS_IMAGE=ubuntu

FROM ${OS_IMAGE}:${OS_VER}
USER root
WORKDIR /ui

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    python3.9 python3.9-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# avoid installing packages while re-build
COPY ./ui/requirements.txt requirements.txt

RUN python3.9 -m venv /root/frontend
RUN . /root/frontend/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    deactivate

COPY ./ui ./
COPY ./utils ./utils/
COPY ./conf ./

COPY ./video_ingest/videos /video_ingest/videos/
COPY ./video_ingest/scene_description /video_ingest/scene_description/

ENTRYPOINT ["bash", "start_ui.sh"]