FROM python:3.8


RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  # python3.8 \
  # python3-pip \
  # python3-setuptools \
  git \
  wget \
  && apt-get clean && rm -rf /var/lib/apt/lists/*
  
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
  
WORKDIR /code

RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
	PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces

# RUN conda install python=3.8

RUN pip install setuptools-rust
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install gradio scikit-image pillow openmim
RUN pip install --upgrade setuptools==69.5.1

WORKDIR /home/user

RUN --mount=type=secret,id=git_token,mode=0444,required=true \
    git clone --branch mmseg-only https://$(cat /run/secrets/git_token)@github.com/NASA-IMPACT/hls-foundation-os.git


WORKDIR hls-foundation-os 

RUN git checkout 9968269915db8402bf4a6d0549df9df57d489e5a

RUN pip install -e .

RUN mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/11.3/1.11.0/index.html

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/code/miniconda/lib"

# Copy the current directory contents into the container at $HOME/app setting the owner to the user

COPY --chown=user . $HOME/app

CMD ["python3", "app.py"]