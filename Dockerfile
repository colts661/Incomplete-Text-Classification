# Base container
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2023.1-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# change to root to update packages
USER root

RUN apt update

# install packages using notebook user
USER jovyan

RUN pip install numpy==1.22.0 threadpoolctl==3.1.0
RUN pip install scipy==1.10.0
RUN pip install transformers[torch] gensim
RUN pip install wordcloud openai backoff

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]