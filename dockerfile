# Use an official image as a base
FROM python:3.10.9

# Set the working directory in the container
WORKDIR /app

# Copy the application code
COPY .  .

RUN pip install  --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the command when the container launches
CMD ["uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"]

