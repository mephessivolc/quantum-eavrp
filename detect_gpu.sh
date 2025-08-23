#!/bin/bash

# detect_gpu.sh
# Gera automaticamente o arquivo .env com configurações para uso de GPU (NVIDIA ou AMD) ou fallback para CPU

ENV_FILE=".env"

echo "🔍 Detectando GPU disponível..."

# Variáveis padrão para CPU (evita erro em docker-compose)
GPU_DRIVER="none"
GPU_COUNT="0"
GPU_CAPABILITIES="compute"         # sempre definir valor padrão
GPU_DEVICES="/dev/null"
NVIDIA_VISIBLE_DEVICES="void"
AMD_VISIBLE_DEVICES="void"

# Verifica NVIDIA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detectada"
    GPU_DRIVER="nvidia"
    GPU_COUNT="all"
    GPU_CAPABILITIES="compute,utility"
    GPU_DEVICES="/dev/nvidia0"
    NVIDIA_VISIBLE_DEVICES="all"

elif command -v rocminfo &> /dev/null; then
    # Verifica se ROCm realmente funciona
    if rocminfo | grep -q "Name:"; then
        echo "✅ AMD GPU compatível com ROCm detectada"
        GPU_DRIVER="amdgpu"
        GPU_COUNT="all"
        GPU_CAPABILITIES="compute"
        GPU_DEVICES="/dev/kfd"
        AMD_VISIBLE_DEVICES="all"
    else
        echo "⚠️ AMD detectada mas não compatível com ROCm. Usando CPU."
    fi
else
    echo "⚠️ Nenhuma GPU compatível detectada. Usando CPU."
    GPU_DRIVER=""
    GPU_COUNT=""
    GPU_CAPABILITIES=""
    GPU_DEVICES=""
    NVIDIA_VISIBLE_DEVICES="void"
    AMD_VISIBLE_DEVICES="void"
fi

# Gera o arquivo .env com segurança
echo "📄 Gerando arquivo .env..."
cat <<EOF > "$ENV_FILE"
# Arquivo gerado automaticamente por detect_gpu.sh

GPU_DRIVER=${GPU_DRIVER}
GPU_COUNT=${GPU_COUNT}
GPU_CAPABILITIES=${GPU_CAPABILITIES}
GPU_DEVICES=${GPU_DEVICES}
NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}
AMD_VISIBLE_DEVICES=${AMD_VISIBLE_DEVICES}
EOF

echo "✅ Arquivo .env criado com sucesso:"
cat "$ENV_FILE"
