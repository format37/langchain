FROM python:3.11

RUN pip install --upgrade pip

copy requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

copy app.py /app.py

CMD ["python", "/app.py"]