ARG OS_VER=22.04
ARG OS_IMAGE=ubuntu

FROM ${OS_IMAGE}:${OS_VER}
USER root
WORKDIR /gateway

#prerequest
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    python3.9 python3.9-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /gateway/

RUN python3.9 -m venv /root/vectorDB
RUN . /root/vectorDB/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    deactivate

ENTRYPOINT ["/root/vectorDB/bin/python3.9","main.py"]
