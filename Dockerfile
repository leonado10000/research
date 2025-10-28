# Base image (Python)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (so this step is cached)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy your actual source code
COPY . .

# Default command
# CMD ["python", "train.py"]
CMD ["python", "eval.py"]