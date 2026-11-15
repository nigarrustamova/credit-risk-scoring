FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
# API:  docker run -p 8000:8000 crs
# Demo: docker run -p 8501:8501 crs python -m streamlit run app/app.py --server.address=0.0.0.0
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
