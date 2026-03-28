%%writefile Dockerfile
FROM python:3.9-slim

# Build tools install karna zaroori hai agar scikit-learn ya pandas install ho rahe hon
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# --no-cache-dir se build fast hota hai aur crash kam hota hai
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables setting
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 7860

# Shell form use karke app run karein
CMD ["streamlit", "run", "app.py"]
