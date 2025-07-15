# Full system Dockerfile for development and testing
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and results
RUN mkdir -p data results logs

# Create non-root user
RUN useradd -m -s /bin/bash fluser && chown -R fluser:fluser /app
USER fluser

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8080 8081 8082

# Default command
CMD ["python", "main.py", "--mode", "full", "--log-level", "INFO"]
