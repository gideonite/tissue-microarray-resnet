# FROM b.gcr.io/tensorflow/tensorflow:latest-devel-gpu
#FROM gcr.io/tensorflow/tensorflow:latest-gpu

# FROM nvidia/cuda:7.5-cudnn4-devel
FROM nvidia/cuda:7.5-cudnn5-devel

MAINTAINER gmd

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
    h5py

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter

# install Tensorflow
ENV TENSORFLOW_VERSION 0.8.0
# RUN pip --no-cache-dir install \
    # http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
    # http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl


# TensorBoard
# EXPOSE 6006

CMD ["/bin/bash"]
