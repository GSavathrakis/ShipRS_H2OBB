FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
WORKDIR /workspace

RUN  apt-get update \
  && apt-get install -y wget git unzip

RUN  pip install -r requirements.txt