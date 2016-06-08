FROM gw000/keras
COPY requirements.txt /usr/src/app/
RUN pip install -r /usr/src/app/requirements.txt
COPY keras_load_datasets.py /opt/keras_load_datasets.py
RUN python /opt/keras_load_datasets.py
COPY app.py /usr/src/app/
COPY network.py /usr/src/app/network.py
COPY templates/results.html /usr/src/app/templates/
COPY templates/social.css /usr/src/app/templates/
COPY templates/index.html /usr/src/app/templates/
RUN mkdir /usr/src/app/results
CMD ["python", "/usr/src/app/app.py"]
