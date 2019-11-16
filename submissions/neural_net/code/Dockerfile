FROM tensorflow/tensorflow:latest-py3

ENV CUDA_VISIBLE_DEVICES = -1

RUN mkdir code
RUN mkdir solution

# Installing requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install nltk packages
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords

# Copy code, data, and models
COPY prediction.py code/prediction.py
COPY test.csv code/test.csv
COPY models/tokenizer.pickle code/tokenizer.pickle
COPY models/keras_try1.h5 code/keras_try1.h5

WORKDIR code
CMD ["python", "prediction.py"]