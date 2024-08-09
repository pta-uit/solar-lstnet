FROM python:3.10

WORKDIR /data-preprocessing

COPY . .

RUN pip install -r requirements.txt

RUN pip install awscli

#ENTRYPOINT ["python", "preprocess_data.py"]

ENTRYPOINT ["python", "main.py"]