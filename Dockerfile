FROM gideonitemd/hal-tf

MAINTAINER gideonitemd

# apt-get
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python-opencv \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# some kind of cv2 nonsense. This maps some kind of raw camera input
# device to `/dev/null`. Otherwise `import cv2` results in an error.
# ln -s /dev/null /dev/raw1394

CMD ["/bin/bash"]

WORKDIR "/mnt/code/"
