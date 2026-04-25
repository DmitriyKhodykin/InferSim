FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости (для Streamlit и Plotly)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Порт Streamlit
EXPOSE 8501

# Команда запуска
CMD ["streamlit", "run", "front/main.py", "--server.port=8501", "--server.address=0.0.0.0"]