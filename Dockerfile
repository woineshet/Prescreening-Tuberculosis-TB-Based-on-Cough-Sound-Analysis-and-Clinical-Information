# Use Python 3.10 (compatible with TensorFlow 2.11+)
FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy all app files into the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set a writable config directory to avoid permission issues
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit

# Expose the port used by Streamlit
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.enableCORS=false"]
