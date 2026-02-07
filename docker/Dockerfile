FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    nano \ 
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    freeglut3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    gdb \
    lldb \
    clang \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/DepthAnything/Depth-Anything-V2
RUN cd Depth-Anything-V2 && pip3 install -r requirements.txt

RUN git clone https://github.com/tauzn-clock/RescalingCompressedSensing
RUN cd RescalingCompressedSensing