FROM python:3.8.6-buster

COPY TaxiFareModel /TaxiFareModel
COPY model.joblib /model.joblib
COPY api /api
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
#RUN pip install Cython
RUN pip install cython==0.28.5
RUN pip install numpy==1.19.2
RUN pip install scikit-learn==0.22
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port=8000 --reload
