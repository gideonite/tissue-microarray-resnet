FROM b.gcr.io/tensorflow/tensorflow:latest-devel-gpu

MAINTAINER gmd

RUN apt-get update && apt-get install -y \
    python-scipy

RUN pip install -U scikit-learn
