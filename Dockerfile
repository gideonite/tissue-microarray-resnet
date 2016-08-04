FROM gideonitemd/hal-tf

MAINTAINER gideonitemd

# apt-get
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python-opencv \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install keras

# set tensorflow as the backend
RUN mkdir ~/.keras && echo {\"epsilon\": 1e-07, \"floatx\": \"float32\", \"backend\": \"tensorflow\"} > ~/.keras/keras.json

CMD ["/bin/bash"]

WORKDIR "/mnt/code/"
