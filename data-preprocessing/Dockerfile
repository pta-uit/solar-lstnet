FROM python:3.10-slim AS builder

WORKDIR /data-preprocessing

COPY requirements.txt ./

RUN pip install -r requirements.txt

######################################
FROM python:3.10-slim

WORKDIR /data-preprocessing

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY . .

RUN pip install --progress-bar off awscli

ENTRYPOINT ["python", "data-preprocessing.py"]