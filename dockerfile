# Imagen base con Python y soporte para PyTorch y OpenCV
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos necesarios
COPY requirements.txt . 
COPY . .

# Instalar paquetes de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Comando por defecto al ejecutar el contenedor
CMD ["python", "yolo_detection.py"]
