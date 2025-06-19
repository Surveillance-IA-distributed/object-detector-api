# Microservicio video-data-ms
Este es un microservicio hecho en python que tiene el objetivo de extraer caracteristicas de frames de videos y almacenarlas en una base de datos PostgreSQL

## Configuracion de variables de entorno
Copia el archivo `.env.template` a `.env` 

```bash
# Contenido del archivo .env.template
# NATS Configuration
NATS_URL=nats://nats:4222
NATS_TIMEOUT=30.0
# Modelo YOLO y configuración
YOLO_MODEL=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5

```

## Compilacion y ejecucion del proyecto
Redirigete al directorio principal del proyecto donde esta el archivo `docker-compose.yml` cuando te clonaste el repositorio [surveillance-core](https://github.com/Surveillance-IA-distributed/surveillance-core) y ejecuta el siguiente comando:
```bash
$ docker compose build
$ docker compose up -d
```