# Use a minimal Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (optional for clarity)
EXPOSE 8000

# Run your Flask app
CMD ["python", "server.py"]
