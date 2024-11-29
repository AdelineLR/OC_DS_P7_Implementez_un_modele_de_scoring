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

# Copy the rest of the application code into the container
COPY app/ /app/

# Expose port 8000 to access the application
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]