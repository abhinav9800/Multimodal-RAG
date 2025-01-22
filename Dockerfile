FROM python:3.9-slim-buster

RUN apt update -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]