FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/ ./proto/
COPY client/ ./client/
COPY data/ ./data/

# Re-compile proto to be safe
RUN python -m grpc_tools.protoc -I./proto --python_out=./proto --grpc_python_out=./proto ./proto/federated.proto
RUN sed -i 's/import federated_pb2 as federated__pb2/from . import federated_pb2 as federated__pb2/' ./proto/federated_pb2_grpc.py

ENTRYPOINT ["python", "-m", "client.client"]
