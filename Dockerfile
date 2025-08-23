
FROM python:slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Instala pacotes básicos e libs matemáticas
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia dependências
COPY requirements.txt .

# Instala pacotes Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 4000

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=4000", "--no-browser", "--allow-root"]
