# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate the model during build
RUN python train_model.py

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
