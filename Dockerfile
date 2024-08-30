# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python packages and AWS CLI from builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/aws-cli /usr/local/aws-cli

# Set up environment
ENV PATH=/root/.local/bin:/usr/local/aws-cli/v2/current/bin:$PATH
ENV PREPROCESSED_DATA=""
ENV MODEL_PATH=""
ENV GPU=-1
ENV FLASK_ENV=production

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Run the application
# CMD ["sh", "-c", "python api.py --preprocessed_data $PREPROCESSED_DATA --model_path $MODEL_PATH --gpu $GPU"]
CMD ["sh", "-c", "python api.py"]