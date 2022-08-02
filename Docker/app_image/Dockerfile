FROM python:3.9

WORKDIR /usr/src/app

COPY Docker/app_image/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

COPY models/label_encoder.sk ./models/

CMD ["python3"]

