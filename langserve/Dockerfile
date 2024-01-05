FROM python:3.11

RUN pip install --upgrade pip

copy requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

copy serve.py /serve.py

CMD ["python", "/serve.py"]