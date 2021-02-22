FROM tensorflow/tensorflow:2.4.1
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app/
RUN pip install .
CMD ["bash", "-c", "python tools/${SCRIPT}.py ${SCRIPT_ARGS}"]
