FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean
    
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (the 'app' folder) into the container
COPY app /app/app

# Copy the tests code (the 'tests' folder) into the container
COPY tests /app/tests

# Expose port 8000 to access the application
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]