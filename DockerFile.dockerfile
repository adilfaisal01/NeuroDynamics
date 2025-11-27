# install pytorch with cuda

FROM pytorch/pytorch:2.9.1-cuda-12.8-cudnn9-runtime

ENV DOCKER_USER_HOME=/root
ENV DEBIAN_FRONTEND=noninteractive

# Work from root
WORKDIR /

# Install utilities
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git python3 python3-pip vim cmake build-essential

RUN pip3 install --upgrade pip 

# copy and install requirements.txt
COPY requirements.txt /deeplearningtransformers/requirements.txt
RUN pip3 install -r /deeplearningtransformers/requirements.txt

#move to project directory
WORKDIR /deeplearningtransformers

COPY . /deeplearningtransformers

ENV output_dir=/deeplearningtransformers/outputs
ENV dataset_dir=/deeplearningtransformers/datasets

COPY ./entrypoint.sh /deeplearningtransformers/entrypoint.sh
RUN   chmod +x /deeplearningtransformers/entrypoint.sh

# default directory
WORKDIR /deeplearningtransformers



