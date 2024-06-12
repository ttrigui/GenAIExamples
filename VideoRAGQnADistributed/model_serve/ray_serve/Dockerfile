ARG OS_VER=22.04
ARG OS_IMAGE=ubuntu

FROM ${OS_IMAGE}:${OS_VER}
USER root
WORKDIR /ray_serve
COPY . /ray_serve

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    python3.9 python3.9-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/root/vectorDB/bin/python3.9:$PATH
ENV PYTHONUNBUFFERED=1
    
RUN python3.9 -m venv /root/ray
RUN . /root/ray/bin/activate && \
    pip install --no-cache-dir -r /ray_serve/requirements.txt && \
    deactivate

ENTRYPOINT ["bash","start_serve.sh"]