FROM morganekmefjord/basesci:latest

RUN mkdir /app
COPY sdk /app/sdk
COPY cli /app/cli
WORKDIR /app
RUN pip install -e cli
RUN pip install -e sdk