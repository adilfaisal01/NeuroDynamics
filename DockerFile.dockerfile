FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /deeplearningtransformers

# Install SSH server + tools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openssh-server \
    vim \
 && rm -rf /var/lib/apt/lists/*

# Set up SSH so we can connect and download results
RUN mkdir /var/run/sshd && \
    echo 'root:runpod' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh runpod_sweep.sh

ENV output_dir=/workspace/outputs
ENV dataset_dir=/deeplearningtransformers/datasets

ENTRYPOINT ["/bin/bash", "/deeplearningtransformers/entrypoint.sh"]