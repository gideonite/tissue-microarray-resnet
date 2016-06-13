# N.B. make sure this matches what is on your system. There is some
# confusion about cudnn versioning. See
# https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html. "cuDNN
# 6.5(v2), 7.0(v3), v5)"
FROM nvidia/cuda:7.5-cudnn3-devel

MAINTAINER gideonitemd

# apt-get
RUN apt-get update && apt-get install -y \
    curl \
    libfreetype6-dev \
    libpng12-dev \
    libzmq3-dev \
    python-scipy \
    python-yaml \
    libhdf5-serial-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install scikit-learn \
    h5py \
    keras

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter

# install Tensorflow
ENV TENSORFLOW_VERSION 0.8.0
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

# TensorBoard
# EXPOSE 6006

CMD ["/bin/bash"]
