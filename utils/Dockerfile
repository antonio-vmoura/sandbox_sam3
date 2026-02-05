# Use a Python 3.12 base image
ARG type
FROM python:3.12 AS base

RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

ENV DATASCIENCE_USER datascience
ENV DATASCIENCE_UID 1000
ENV HOME /home/$DATASCIENCE_USER

# Copy the requirements file (if you have one)
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS run-type-local
# nothing to see here

FROM base AS run-type-remote
COPY sam3_video_blur.py .
CMD ["python", "sam3_video_blur.py"]

FROM run-type-${type} AS final