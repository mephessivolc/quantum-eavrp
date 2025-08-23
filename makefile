# Nome do container
CONTAINER_NAME=quantum-lab

# Diretório de notebooks
NOTEBOOK_DIR=./notebooks

# Detecta GPU e cria .env
detect-gpu:
	@chmod +x detect_gpu.sh
	@./detect_gpu.sh

# Cria o ambiente (detecta GPU + constrói container)
create:
	@echo "🔍 Detectando GPU e configurando ambiente..."
	@bash detect_gpu.sh
	@echo "🚀 Iniciando container..."
	docker compose build
# Executa o container (deve ter sido criado antes)
run:
	@echo "🔍 Detectando GPU..."
	@bash detect_gpu.sh
	@if [ "$$(grep GPU_DRIVER .env | cut -d= -f2)" = "nvidia" ] || \
        [ "$$(grep GPU_DRIVER .env | cut -d= -f2)" = "amdgpu" ]; then \
		echo "🚀 GPU detectada. Usando docker-compose.gpu.yml"; \
		docker compose -f docker-compose.gpu.yml up --build; \
	else \
		echo "⚙️ Nenhuma GPU compatível detectada. Usando CPU."; \
		docker compose -f docker-compose.yml up --build; \
	fi
# Para o container
stop:
	docker compose down

# Abre terminal interativo dentro do container
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Remove container, cache e volumes (mas mantém imagem base)
clean:
	docker compose down -v --rmi all --remove-orphans

# Remove tudo (⚠️ imagens, volumes, containers)
nuke:
	docker system prune -a --volumes -f
