# Use official Python image or your base image
FROM python:3.9-slim

# Set working directory
WORKDIR /home/user/app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set Streamlit config path
ENV STREAMLIT_CONFIG_FILE=/home/user/app/src/.streamlit/config.toml

# Run the Streamlit app
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]


