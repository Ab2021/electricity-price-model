# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and data into the container
COPY src/ /app/src/
COPY data/ /app/data/

# Set the command to run the training script
CMD ["python", "src/train.py"]