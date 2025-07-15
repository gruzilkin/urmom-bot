FROM python:3.13-slim

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy web-specific requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy web application code
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/

# Expose port
EXPOSE 8000

# Command to run the web app
CMD ["python", "-m", "src.main"]