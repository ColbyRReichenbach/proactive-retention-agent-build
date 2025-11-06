# Start from an official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first and install dependencies
# This is cached by Docker, so it's faster
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your application code
COPY . .

# Expose the port that uvicorn will run on
EXPOSE 8000

# Command to run the application
# Uvicorn is the server that runs your FastAPI app
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]