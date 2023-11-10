FROM python:3.9.16-slim-buster
WORKDIR /server
COPY requirements.txt /server
RUN pip3 install -r requirements.txt
COPY . /server
CMD ["flask", "run"]
