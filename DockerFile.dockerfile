FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /deeplearningtransformers

# Only what's needed beyond the base image
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    vim \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh runpod_sweep.sh

ENV output_dir=/deeplearningtransformers/outputs
ENV dataset_dir=/deeplearningtransformers/datasets

ENTRYPOINT ["/bin/bash", "/deeplearningtransformers/entrypoint.sh"]