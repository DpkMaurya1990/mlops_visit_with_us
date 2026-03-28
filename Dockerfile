FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY . .

# Set Port and Host
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 7860

# Hinglish: Isse container turant kill nahi hoga agar streamlit me koi choti error aaye
CMD ["sh", "-c", "streamlit run app.py --server.port=7860 --server.address=0.0.0.0"]
