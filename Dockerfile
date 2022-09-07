FROM python:3.8.12-buster
COPY job_predictor /job_predictor
COPY model/bert_model_22.sav /model/bert_model_22.sav
COPY model/all_corpus_embed_22.sav /model/all_corpus_embed_22.sav
COPY data /data
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn job_predictor.api.fast:app --host 0.0.0.0
