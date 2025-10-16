# Dockerfile para DESARROLLO
# Imagen base con Python y soporte para PyTorch y OpenCV
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema (runtime + herramientas de desarrollo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Librerías runtime necesarias para OpenCV
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt primero (mejor cache)
COPY requirements.txt .

# Instalar PyTorch CPU-only PRIMERO (igual que en producción)
# Esto reduce el tamaño de ~12GB a ~2-3GB también en desarrollo
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1+cpu torchvision==0.20.1+cpu

# Instalar requirements.txt (sin ultralytics todavía)
RUN pip install --no-cache-dir -r requirements.txt

# Instalar ultralytics sin dependencias (usa el torch ya instalado)
RUN pip install --no-cache-dir --no-deps ultralytics

# OPCIONAL para desarrollo: Instalar herramientas de debugging
# Descomenta si las necesitas:
# RUN pip install --no-cache-dir ipython ipdb pytest pytest-asyncio

# Copiar el resto del código (al final para mejor cache)
COPY . .

# Variables de entorno para desarrollo
ENV ENV=development

# Comando por defecto al ejecutar el contenedor
CMD ["python", "main.py"]
