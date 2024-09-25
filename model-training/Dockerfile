FROM python:3.10-slim AS builder

WORKDIR /model-training

COPY requirements.txt ./

RUN pip install --progress-bar off -r requirements.txt

#########################
#########################

FROM python:3.10-slim

WORKDIR /model-training

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY . .

RUN pip install --progress-bar off awscli

ENTRYPOINT ["python", "main.py"]