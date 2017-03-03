FROM tensorflow/tensorflow:1.0.0-devel
RUN mkdir /language_model && mkdir /language_model/models && mkdir /language_model/models/wiki
COPY config.py /language_model
COPY vocab.py /language_model
COPY data_utils.py /language_model
COPY language_model.py /language_model
COPY model_server.py /language_model
COPY model_utils.py /language_model
COPY text_generator.py /language_model
COPY models/wiki /language_model/models/wiki
RUN pip install flask && pip install flask-cors
WORKDIR /language_model
CMD ["python", "model_server.py"]