# Use the official slim Python image as the base image
FROM python:slim

# Set environment variables to prevent Python from writing .pyc files 
# and to ensure output is flushed directly to the terminal
ENV PYTHONDONTWRITEBYTECODE = 1\
    PYTHONUNBUFFERED = 1

# Set the working directory inside the container
WORKDIR /app

# Update the package list and install required system dependencies
# - libgomp1: Required for certain machine learning libraries like XGBoost
# Clean up the apt cache to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files from the host to the container's working directory
COPY . .

# Install Python dependencies listed in the project
# The `-e .` flag installs the package in editable mode
# The `--no-cache-dir` flag prevents caching to reduce image size
RUN pip install ---no-cache-dir -e .

# Run the training pipeline script to prepare the model
RUN python pipeline/training_pipeline.py

# Expose port 5000 for the application to be accessible
EXPOSE 5000

# Set the default command to run the application
CMD ["python", "application.py"]