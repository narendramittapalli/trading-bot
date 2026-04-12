FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for logs and state (Railway volume should be mounted here)
RUN mkdir -p logs

# Run the adaptive daily scheduler
CMD ["python", "main.py", "run"]
