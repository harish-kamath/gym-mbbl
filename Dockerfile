# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

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

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .
RUN pip install --upgrade pytorch_lightning
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python

RUN conda install -y -c nvidia/label/cuda-11.3.0 cuda-nvcc
RUN conda install -y -c conda-forge gcc
RUN conda install conda-libmamba-solver
RUN conda config --set experimental_solver libmamba
RUN conda install -y -c conda-forge gxx_linux-64=9.5.0
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=6.0  pip install git+https://github.com/facebookresearch/xformers

ADD . .
RUN pip install --no-deps -e .
CMD python3 -u server.py
