# src/database_handler
import asyncpg
import os
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DatabaseHandler:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '123456'),
            'database': os.getenv('DB_NAME', 'videosdb')
        }
        self.pool = None
    
    async def initialize(self):
        """Inicializar pool de conexiones."""
        try:
            self.pool = await asyncpg.create_pool(**self.db_config, max_size=10)
            logger.info("Base de datos inicializada correctamente")
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
            raise
    
    async def insert_detection(self, detection_data: Dict[str, Any]) -> bool:
        """
        Insertar una detección en la tabla Detections.
        
        Args:
            detection_data: Diccionario con los datos de la detección:
                - stream_session_id (int): ID de la sesión de streaming
                - clase (str): Clase del objeto detectado
                - x, y, ancho, alto (float): Coordenadas del bounding box
                - confianza (float): Nivel de confianza de la detección
                - milisegundos (float): Timestamp en milisegundos
                - detection_time (str): Tiempo de detección (formato HH:MM:SS)
                - color (str, optional): Color promedio
                - color_superior (str, optional): Color de la parte superior
                - color_inferior (str, optional): Color de la parte inferior
        
        Returns:
            bool: True si se insertó correctamente, False en caso contrario
        """
        try:
            async with self.pool.acquire() as connection:
                await connection.execute("""
                    INSERT INTO "Detections" (
                        "streamSessionId", clase, x, y, ancho, alto, confianza, 
                        milisegundos, "detectionTime", color, "colorSuperior", "colorInferior",
                        "createdAt", "updatedAt"
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                detection_data.get('stream_session_id'),
                detection_data.get('clase', 'unknown'),
                float(detection_data.get('x', 0)),
                float(detection_data.get('y', 0)),
                float(detection_data.get('ancho', 0)),
                float(detection_data.get('alto', 0)),
                float(detection_data.get('confianza', 0.0)),
                float(detection_data.get('milisegundos', 0)),
                detection_data.get('detection_time', datetime.now().strftime('%H:%M:%S')),
                detection_data.get('color'),
                detection_data.get('color_superior'),
                detection_data.get('color_inferior'),
                datetime.now(),  # createdAt
                datetime.now()   # updatedAt
                )
            
            logger.info(f"Detección insertada: {detection_data.get('clase')} en stream session {detection_data.get('stream_session_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error insertando detección: {e}")
            return False
    
    async def insert_multiple_detections(self, detections: List[Dict[str, Any]]) -> int:
        """
        Insertar múltiples detecciones en una transacción.
        
        Args:
            detections: Lista de diccionarios con datos de detecciones
        
        Returns:
            int: Número de detecciones insertadas exitosamente
        """
        if not detections:
            return 0
        
        inserted_count = 0
        
        try:
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    for detection_data in detections:
                        try:
                            await connection.execute("""
                                INSERT INTO "Detections" (
                                    "streamSessionId", clase, x, y, ancho, alto, confianza, 
                                    milisegundos, "detectionTime", color, "colorSuperior", "colorInferior",
                                    "createdAt", "updatedAt"
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                            """,
                            detection_data.get('stream_session_id'),
                            detection_data.get('clase', 'unknown'),
                            float(detection_data.get('x', 0)),
                            float(detection_data.get('y', 0)),
                            float(detection_data.get('ancho', 0)),
                            float(detection_data.get('alto', 0)),
                            float(detection_data.get('confianza', 0.0)),
                            float(detection_data.get('milisegundos', 0)),
                            detection_data.get('detection_time', datetime.now().strftime('%H:%M:%S')),
                            detection_data.get('color'),
                            detection_data.get('color_superior'),
                            detection_data.get('color_inferior'),
                            datetime.now(),  # createdAt
                            datetime.now()   # updatedAt
                            )
                            inserted_count += 1
                            
                        except Exception as e:
                            logger.error(f"Error insertando detección individual: {e}")
                            # Continúa con las siguientes detecciones
                    
            logger.info(f"Insertadas {inserted_count} de {len(detections)} detecciones")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error en transacción de múltiples detecciones: {e}")
            return 0
    
    async def close(self):
        """Cerrar pool de conexiones."""
        if self.pool:
            await self.pool.close()
            logger.info("Pool de conexiones cerrado")