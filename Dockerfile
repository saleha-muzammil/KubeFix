FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY kubefix.py ./kubefix.py
ENV GEMINI_MODEL=gemini-2.0-flash
EXPOSE 8080
CMD ["uvicorn", "kubefix:app", "--host", "0.0.0.0", "--port", "8080"]

