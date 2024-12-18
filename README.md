# Running Resemble Enhance in Triton Inference Server

## Installation

Run these command in triton container to install dependencies

```bash
apt update && apt install ffmpeg git-lfs -y
pip install resemble-enhance --upgrade
```

## Running 
Build the docker container like this:
```bash
docker build -t resemble-enhancer-triton-container .
```

After which run the container:

```bash
docker run -it --shm-size 1gb --network host --name Resemble-Enhancer-Triton-Server --gpus device=0 resemble-enhancer-triton-container
```

And start the server:

```bash
tritonserver --model-repository /models --http-port=8520 --grpc-port=8521 --metrics-port=8522
```

## Testing by client:

Install requirements:

```bash
pip install -r requirements.txt
```

And run the client script:

```
python -m clients.client --triton-address 127.0.0.1 --triton-port 8520 --input-file [your_input_file].wav
```

**Ensure** that your client environment can reach the Triton Server container.

## Useful links

Original repository: https://github.com/resemble-ai/resemble-enhance/tree/main

Triton Guides: https://github.com/triton-inference-server/tutorials
