FROM python:3.13-slim

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code from src directory
COPY src .

# Command to run the bot
CMD ["python", "app.py"]