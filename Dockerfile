# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies file
COPY requirements.txt .
COPY flagged.csv .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the monitor script into the container
COPY alerta_erro_malha.py ./monitor.py

# Run the script
CMD ["python", "monitor.py"]

