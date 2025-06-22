FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for Claude Code CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs

# Install Claude Code CLI via npm
RUN npm install -g @anthropic-ai/claude-code

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code from src directory
COPY bot/ .

# Command to run the bot
CMD ["python", "app.py"]