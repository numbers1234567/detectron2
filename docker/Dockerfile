# Import base Nvidia image with ubuntu 22.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set build argumetns
# Build arg for your GitHub access key
ARG ACCESS_KEY

# Optional build arg for specifying the branch to clone
ARG GIT_BRANCH=main

# Set enviroment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=host.docker.internal:0.0

# Update the image
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends apt-utils

# Get python 3.10
RUN apt-get install -y python3-pip

# Install needed apt packages
RUN apt-get install -y apt-transport-https \
    curl \
    gedit \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    locales \
    openssh-client \
    software-properties-common \
    git-lfs

RUN apt-get install -y apt-transport-https \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcomposite1 \
    libxcursor1 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libglib2.0-0 \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    psmisc \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-shm0 \
    libxcb-glx0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xkb1 \
    libx11-xcb1 \
    libxcb1 \
    libgl1-mesa-glx \
    libxrender1 \
    libxi6 \
    libxkbcommon-x11-0 \
    libsm6 \
    libxcb-xinerama0-dev

ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} detectron2_user -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER detectron2_user
WORKDIR /home/detectron2_user

ENV PATH="/home/detectron2_user/.local/bin:${PATH}"

# Install Python packages
RUN pip3 install --user torch==2.0.1 torchvision==0.15.2 -f https://download.pytorch.org/whl/cu117/torch_stable.html
RUN pip3 install --user 'git+https://github.com/facebookresearch/fvcore'
	
RUN git clone --branch ${GIT_BRANCH} --single-branch https://github.com/numbers1234567/detectron2.git
WORKDIR /home/detectron2_user/detectron2

RUN git lfs pull

# Setup detectron2 with CUDA
ENV FORCE_CUDA="1"

# Ideally we should know which architecture to target to reduce build times
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Install Python packages
RUN pip3 install -e .
RUN pip3 install -r requirements.txt