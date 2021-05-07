# syntax=docker/dockerfile:experimental
FROM python:3.7-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working diretory
COPY ./neural_rock/ ./neural_rock
COPY ./app ./app
COPY ./api ./api
COPY ./data ./data
COPY ./requirements.txt ./requirements.txt

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt

# Run the web api
EXPOSE 80
EXPOSE 8000
ENV PYTHONPATH /repo
CMD /bin/bash