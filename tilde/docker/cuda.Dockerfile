# tilde-server with CUDA. Build from the repository root:
#   docker build -f tilde/docker/cuda.Dockerfile -t tilde-server:cuda .
# Run (downloads the GGUF on first start, picks the preset by free VRAM):
#   docker run --gpus all -p 8080:8080 -v tilde-cache:/root/.cache tilde-server:cuda
ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1
ARG BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# Turing, Ampere, Ada, Hopper, Blackwell (consumer and data center)
ARG CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;120-real;121-virtual"

RUN apt-get update && apt-get install -y build-essential cmake git libssl-dev libgomp1 curl

WORKDIR /app
COPY . .

RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --config Release -j$(nproc) --target tilde-server llama-server

RUN mkdir -p /app/out && cp build/bin/tilde-server build/bin/llama-server /app/out && \
    find build -name "*.so*" -exec cp -P {} /app/out \;

FROM ${BASE_CUDA_RUN_CONTAINER}

RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/out/ /app/
WORKDIR /app
ENV LD_LIBRARY_PATH=/app LLAMA_ARG_HOST=0.0.0.0 LLAMA_ARG_PORT=8080

LABEL org.opencontainers.image.title="tilde-server" \
      org.opencontainers.image.description="Tilde decision model server (/v1/systemone)" \
      org.opencontainers.image.source="https://github.com/resident-labs/tilde.cpp"

EXPOSE 8080
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/tilde-server" ]
