FROM gideonitemd/hal-tf

MAINTAINER gideonitemd

# apt-get
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python-opencv \
    wget \
    libjpeg-dev \
    libpng-dev \       
    libtiff-dev \
    cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# some kind of cv2 nonsense. This maps some kind of raw camera input
# device to `/dev/null`. Otherwise `import cv2` results in an error.
# ln -s /dev/null /dev/raw1394

RUN OPENCV_VERSION=3.1.0 && \
    wget -q -O - https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - && \
    cd opencv-${OPENCV_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-${OPENCV_VERSION} . && \
    make -j"$(nproc)" install && \
    rm -rf /opencv-${OPENCV_VERSION}

RUN mv -v /usr/local/opencv-3.1.0/lib/python2.7/dist-packages/cv2.so /usr/local/lib/python2.7/site-packages/

ENV PYTHONPATH $PYTHONPATH:/usr/local/lib/python2.7/site-packages

CMD ["/bin/bash"]

WORKDIR "/mnt/code/"
