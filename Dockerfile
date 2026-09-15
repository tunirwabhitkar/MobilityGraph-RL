FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Command to run both services (using a simple shell script or just setting the default to API)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
