ARG OS_VER=22.04
ARG OS_IMAGE=ubuntu

FROM ${OS_IMAGE}:${OS_VER}
USER root
WORKDIR /ray_serve
COPY . /ray_serve

#prerequest
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    cmake \
    build-essential \
    curl \
    numactl \
    gnupg2 \
    git \
    software-properties-common \
    pkg-config \
    libva-dev \
    vim \
    wget \
    ocl-icd-opencl-dev \
    zip \
    gpg \
    ca-certificates \
    gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/miniconda

ENV PATH=/opt/miniconda/bin:$PATH

RUN conda update -n base -c defaults conda && \
    conda create -n ray_serve python=3.9 -y

ENV PATH=/opt/miniconda/envs/ray_serve/bin:$PATH
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir -r /ray_serve/requirements.txt

ENTRYPOINT ["bash","start_serve.sh"]