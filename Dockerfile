FROM python:3.6
RUN apt-get -y update
RUN apt-get install -y --fix-missing cmake	
COPY . ./app
WORKDIR /app
RUN pip install torch torchvision
RUN pip install -r requirements.txt
RUN bash weights/download_weights.sh
