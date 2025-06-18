# main.py
import asyncio
import logging
import os
from dotenv import load_dotenv
from src.microservice import ObjectDetectionMicroservice

# Cargar variables de entorno
load_dotenv()

# Configurar logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Función principal para iniciar el microservicio."""
    try:
        nats_url = os.getenv('NATS_URL', 'nats://localhost:4222')
        service_name = os.getenv('SERVICE_NAME', 'object-detection-service')
        
        microservice = ObjectDetectionMicroservice(nats_url=nats_url, service_name=service_name)
        await microservice.start()
    except KeyboardInterrupt:
        logging.info("Microservicio detenido por el usuario")
    except Exception as e:
        logging.error(f"Error en el microservicio: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 