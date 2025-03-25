FROM python:3.10

WORKDIR /app

COPY req.txt .

RUN pip install --no-cache-dir -r req.txt

COPY . .

EXPOSE 9052

CMD ["streamlit", "run", "main.py", "--server.port=9052", "--server.address=0.0.0.0"]