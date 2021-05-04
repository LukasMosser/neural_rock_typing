# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
FROM python:3.7-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working diretory
COPY ./neural_rock/ ./neural_rock
COPY ./app ./app
COPY ./data/models ./data/models
COPY ./data/Images_PhD_Miami/img ./data/Images_PhD_Miami/img
COPY ./data/Images_PhD_Miami/ROI ./data/Images_PhD_Miami/ROI

# Install Python dependencies
RUN set -ex && pip3 install -r app/requirements.txt

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 -m panel serve /repo/app/my_app.py