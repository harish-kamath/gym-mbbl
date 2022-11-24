# Must use a Cuda version 11+
FROM nvcr.io/nvidia/pytorch:22.08-py3

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git && apt-get install -y curl

# copy self in
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN git lfs install

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Add your huggingface auth key here
ENV HF_AUTH_TOKEN=your_token

# Add your custom app code, init() and inference()
ADD app.py .
RUN pip install --upgrade pytorch_lightning
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST pip install git+https://github.com/facebookresearch/xformers@51dd119#egg=xformers

# RUN conda install -y -c nvidia/label/cuda-11.3.0 cuda-nvcc
# RUN conda install -y -c conda-forge gcc
# RUN conda install conda-libmamba-solver
# RUN conda config --set experimental_solver libmamba
# RUN conda install -y -c conda-forge gxx_linux-64=9.5.0
# RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=6.0  pip install git+https://github.com/facebookresearch/xformers

ADD . .
RUN pip install --no-deps -e .

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

CMD python3 -u server.py
