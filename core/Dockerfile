FROM python:3.12-slim
LABEL authors="lazarevtill"

WORKDIR /opt/morgan

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /opt/morgan/data/logs

# Expose API port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]