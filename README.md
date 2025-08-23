# **Quantum VRP Lab** 

*Ambiente de desenvolvimento para pesquisa e simulação de algoritmos
quânticos e clássicos aplicados ao problema de roteamento de veículos
elétricos autônomos (EA-VRP).*

------------------------------------------------------------------------

## **📌 Objetivo**

Este projeto fornece um **ambiente unificado** para pesquisa e
desenvolvimento de soluções para o **Vehicle Routing Problem** (VRP)
aplicado a **frotas de veículos elétricos autônomos**.

Ele suporta: - **Simulações clássicas** para benchmarking. -
**Algoritmos quânticos** com **Pennylane** e/ou **Qiskit**. - Execução
**otimizada via GPU** (NVIDIA ou AMD/ROCm) **quando disponível**. -
**Fallback automático para CPU** caso nenhuma GPU compatível seja
detectada.

Esse ambiente será a **base de código** para todos os experimentos e
artigos do doutorado.

------------------------------------------------------------------------

## **📂 Estrutura do Projeto**

    quantum-vrp-lab/
    ├── Dockerfile                # Ambiente base Python
    ├── docker-compose.yml        # Configuração padrão (CPU)
    ├── docker-compose.gpu.yml    # Configuração avançada (GPU)
    ├── detect_gpu.sh             # Script para detectar NVIDIA, AMD ou fallback CPU
    ├── Makefile                  # Comandos principais do projeto
    ├── requirements.txt          # Dependências Python
    ├── notebooks/                # Jupyter Notebooks para experimentos
    └── README.md                 # Este arquivo

------------------------------------------------------------------------

## **⚡ Pré-requisitos**

-   **Docker** ≥ 24.x\
-   **Docker Compose** ≥ 2.x\
-   **Make** ≥ 4.x\
-   (Opcional) **NVIDIA Container Toolkit** se for utilizar GPUs NVIDIA\
-   (Opcional) **ROCm** configurado para GPUs AMD compatíveis

------------------------------------------------------------------------

## **🚀 Como usar**

### **1. Construir o ambiente**

Use o comando:

``` bash
make create
```

### **2. Iniciar o ambiente**

Use o comando:

``` bash
make run
```
O processo executará:

1.  **Detecta automaticamente** se há GPU NVIDIA ou AMD compatível
    (`detect_gpu.sh`).
2.  **Configura o arquivo `.env`** com as variáveis necessárias.
3.  **Seleciona o docker-compose correto**:
    -   Se houver **GPU NVIDIA compatível** → usa **CUDA**.
    -   Se houver **GPU AMD com ROCm** → usa **ROCm**.
    -   Se não houver GPU → roda com **CPU**.

------------------------------------------------------------------------

### **2. Estrutura dos comandos `make`**

  -----------------------------------------------------------------------
  Comando           Descrição
  ----------------- -----------------------------------------------------
  `make run`        Sobe o ambiente detectando GPU ou caindo para CPU
                    automaticamente.

  `make stop`       Para e remove os containers.

  `make rebuild`    Recria a imagem e reinstala dependências do zero.

  `make logs`       Exibe os logs do container principal.

  `make bash`       Abre um terminal dentro do container para execução
                    interativa.

  `make clean`      Remove containers, volumes, redes e arquivos
                    temporários.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## **🧠 Tecnologias utilizadas**

-   **Python 3.11** (via Docker)
-   **Pennylane** → simulações quânticas e híbridas
-   **Qiskit** → simulações clássicas e quânticas
-   **NumPy**, **Matplotlib**, **SciPy** → manipulação e visualização de
    dados
-   **Jupyter Lab** → ambiente para experimentos interativos
-   **Docker Compose** → orquestração de CPU/GPU automaticamente

------------------------------------------------------------------------

## **💡 Sobre o projeto**

Este repositório serve como **base de código do doutorado** e será
utilizado para:

-   Modelar **VRPs complexos** considerando restrições energéticas.
-   Implementar algoritmos **quânticos** e **híbridos**.
-   Comparar desempenho entre **execuções clássicas** e
    **quantum-inspired**.
-   Criar benchmarks replicáveis para artigos científicos.

------------------------------------------------------------------------

## **📜 Licença**

Este projeto está sob licença **MIT**.\
Sinta-se livre para utilizar e adaptar para fins acadêmicos.

------------------------------------------------------------------------

## **👨‍🔬 Autor**

**Clovis Aparecido Caface Filho**\
Programa de Doutorado em Ciência da Computação\
Universidade Federal do ABC - UFABC
Orientador: Raphael de Camargo Yokoingawa
