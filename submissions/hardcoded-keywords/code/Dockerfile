FROM python:3.7

RUN mkdir code
RUN mkdir solution

COPY prediction.py code/prediction.py
COPY test.csv code/test.csv
WORKDIR code

CMD ["python", "prediction.py"]

