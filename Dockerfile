# Base container
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2023.1-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# change to root to update packages
USER root

RUN apt update

# install packages using notebook user
USER jovyan

RUN pip install numpy==1.22.0
RUN pip install gensim

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]