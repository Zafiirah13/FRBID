# syntax = docker/dockerfile:1.0-experimental
# This Dockerfile requires BuildKit support (Docker >= 18.09)
# That will work only if we use Tensorflow >= 1.13
# Anything lower than that - use 9.0 16.04
# We have to use TF 1.x - quite a few breaking changes in 2.x
FROM nvidia/cuda:10.0-base-ubuntu18.04

ENV CUDNN_VERSION 7.6.5.32

RUN echo 'deb http://us.archive.ubuntu.com/ubuntu trusty main multiverse' >> /etc/apt/sources.list && \
    apt update && \
    apt upgrade -y && \
    apt install -y cuda-cudart-$CUDA_PKG_VERSION \
    cuda-cublas-$CUDA_PKG_VERSION \
    cuda-cufft-$CUDA_PKG_VERSION \
    libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
    cuda-curand-$CUDA_PKG_VERSION \
    cuda-cusolver-$CUDA_PKG_VERSION \
    cuda-cusparse-$CUDA_PKG_VERSION 

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

WORKDIR /software
COPY frbid_cuda_10.pip /software/frbid_cuda_10.pip 

RUN apt install -y openssh-client && \
    mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN --mount=type=ssh,id=gitssh \
    apt install -y git  && \ 
    git clone git@github.com:Zafiirah13/FRBID.git && \ 
    cd FRBID && git checkout standalone && \
    apt -y purge git && apt autoremove -y

RUN apt update && \
    apt install -y git python3 \
    python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install -r /software/frbid_cuda_10.pip 

 
