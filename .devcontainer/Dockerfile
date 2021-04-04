FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3
#FROM nvcr.io/nvidia/pytorch:21.02-py3
WORKDIR /home/workspace

# install python
#RUN apt-get update
COPY requirements.txt ${PWD}
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ユーザーを作成
ARG DOCKER_UID=1000
ARG DOCKER_USER=fujimotoat
ARG DOCKER_PASSWORD=inc0124?
RUN useradd -m --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} \
&& echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd

# 作成したユーザーに切り替える
USER ${DOCKER_USER}