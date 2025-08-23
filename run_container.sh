#!/bin/bash

# Detecta a GPU
source detect-gpu.sh

echo "Usando driver: $GPU_DRIVER"
echo "Dispositivos: $GPU_DEVICES"

# Executa o container
docker-compose up --build