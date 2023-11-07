FROM python:3.10

WORKDIR /api

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Load model and encoder
COPY save/rf_pipeline.pkl ./
COPY save/Encode.pkl ./

CMD ["uvicorn", "main:app", "--reload", "--workers", "1", "--host", "0.0.0.0", "--port", "8000"]