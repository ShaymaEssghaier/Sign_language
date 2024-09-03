FROM python:3.12-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    apt-get clean

# Install the dependancies
COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt

# Will execute nltk.download('wordnet')
COPY post_install.py .
RUN python post_install.py

# Copy the code files
COPY src /

# Listen to port 5000 (the default port of flask)
EXPOSE 5000

# Define the working dir in the contener
WORKDIR /

# Command to start the app
CMD ["python", "main.py"]
