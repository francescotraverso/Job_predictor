FROM python:3.8.12-buster
COPY job_predictor /job_predictor
COPY model /model
COPY data /data
COPY chunkers /chunkers
COPY corpora /corpora
COPY taggers /taggers
COPY tokenizers /tokenizers
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn job_predictor.api.fast:app --host 0.0.0.0 --port $PORT
