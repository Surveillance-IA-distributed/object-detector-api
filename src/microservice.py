# src/microservice.py
import asyncio
import json
import logging
import base64
import os
from typing import List, Dict, Any, Optional
import nats
from nats.errors import TimeoutError
import cv2
import numpy as np
from dotenv import load_dotenv
from .yolo_detector import YOLOObjectDetector
from .response_handler import ResponseHandler

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)

class ObjectDetectionMicroservice:  
    def __init__(self, nats_url: str = None, service_name: str = None):
        self.nats_url = nats_url or os.getenv('NATS_URL', 'nats://localhost:4222')
        self.service_name = service_name or os.getenv('SERVICE_NAME', 'object-detection-service')
        self.timeout = float(os.getenv('NATS_TIMEOUT', '30.0'))
        self.max_frames_per_batch = int(os.getenv('MAX_FRAMES_PER_BATCH', '100'))
        self.nc = None
        self.yolo_detector = YOLOObjectDetector()
        self.response_handler = ResponseHandler()
        
    async def start(self):
        """Iniciar el microservicio y conectar a NATS."""
        try:
            # Conectar a NATS
            self.nc = await nats.connect(self.nats_url)
            logger.info(f"Conectado a NATS en {self.nats_url}")
            
            # Suscribirse a múltiples topics
            await self.nc.subscribe("object_detection.analyze_frames", cb=self.handle_frame_analysis)
            await self.nc.subscribe("object_detection.analyze_single_frame", cb=self.handle_single_frame_analysis)
            
            logger.info(f"Microservicio {self.service_name} iniciado. Esperando mensajes...")
            
            # Mantener el servicio corriendo
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error al iniciar el microservicio: {e}")
            raise
        finally:
            if self.nc:
                await self.nc.close()
    
    async def handle_single_frame_analysis(self, msg):
        """Manejar análisis de un solo frame."""
        try:
            logger.info("Recibido mensaje para análisis de frame único")
            
            # Decodificar el mensaje
            data = json.loads(msg.data.decode())
            
            # Validar estructura del mensaje para frame único
            if not self._validate_single_frame_message(data):
                await self._send_simple_error_response(msg, "Estructura de mensaje inválida para frame único")
                return
            
            # Procesar frame único
            result = await self._process_single_frame(data)
            
            # Imprimir resultado
            self._print_single_frame_result(result)
            
            # Simular envío a base de datos
            await self.send_results_database(result, 'single_frame')
            
            # Enviar respuesta simple al NATS
            await self._send_simple_success_response(msg, "Frame procesado exitosamente")
            
        except Exception as e:
            logger.error(f"Error procesando frame único: {e}")
            await self._send_simple_error_response(msg, f"Error en procesamiento: {str(e)}")
    
    async def handle_frame_analysis(self, msg):
        """Manejar análisis de múltiples frames."""
        try:
            logger.info("Recibido mensaje para análisis de frames múltiples")
            
            # Decodificar el mensaje
            data = json.loads(msg.data.decode())
            
            # Validar estructura del mensaje
            if not self._validate_frame_message(data):
                await self._send_simple_error_response(msg, "Estructura de mensaje inválida")
                return
            
            # Verificar límite de frames
            if len(data['frames']) > self.max_frames_per_batch:
                await self._send_simple_error_response(
                    msg, 
                    f"Demasiados frames. Máximo permitido: {self.max_frames_per_batch}"
                )
                return
            
            # Procesar frames
            results = await self._process_frames(data)
            
            # Imprimir resultados de detección
            self._print_detection_results(results)
            
            # Simular envío a base de datos
            await self.send_results_database(results, 'multiple_frames')
            
            # Enviar respuesta simple al NATS
            await self._send_simple_success_response(msg, f"Lote de {results['total_frames']} frames procesado exitosamente")
            
        except Exception as e:
            logger.error(f"Error procesando frames: {e}")
            await self._send_simple_error_response(msg, f"Error en procesamiento: {str(e)}")
    
    async def send_results_database(self, results: Dict[str, Any], analysis_type: str):
        """Simular envío de resultados a base de datos."""
        print(f"\n{'🗄️'*20}")
        print(f"📊 SIMULANDO ENVÍO A BASE DE DATOS")
        print(f"{'🗄️'*20}")
        print(f"Tipo de análisis: {analysis_type}")
        print(f"Timestamp: {self.response_handler._get_timestamp()}")
        
        if analysis_type == 'single_frame':
            print(f"Datos del frame único:")
            print(f"  - Fuente: {results.get('source', 'N/A')}")
            print(f"  - Timestamp: {results.get('timestamp', 'N/A')}")
            print(f"  - Objetos detectados: {results.get('detection_count', 0)}")
            print(f"  - Estado: {'Exitoso' if 'error' not in results else 'Error'}")
            
            if results.get('detections'):
                print(f"  - Clases detectadas:")
                class_counts = {}
                for detection in results['detections']:
                    class_name = detection['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                for class_name, count in class_counts.items():
                    print(f"    • {class_name}: {count}")
        
        elif analysis_type == 'multiple_frames':
            print(f"Datos del lote de frames:")
            print(f"  - Video: {results.get('video_name', 'N/A')}")
            print(f"  - Total frames: {results.get('total_frames', 0)}")
            print(f"  - Frames exitosos: {results.get('successful_frames', 0)}")
            print(f"  - Frames con error: {results.get('failed_frames', 0)}")
            print(f"  - Total objetos detectados: {results['summary']['total_objects_detected']}")
            print(f"  - Frames con objetos: {results['summary']['frames_with_objects']}")
            print(f"  - Promedio objetos/frame: {results['summary']['average_objects_per_frame']:.2f}")
            
            # Contar clases detectadas en todo el lote
            all_classes = {}
            for frame_result in results.get('detections', []):
                if frame_result.get('detections'):
                    for detection in frame_result['detections']:
                        class_name = detection['class']
                        all_classes[class_name] = all_classes.get(class_name, 0) + 1
            
            if all_classes:
                print(f"  - Distribución de clases detectadas:")
                for class_name, count in all_classes.items():
                    print(f"    • {class_name}: {count}")
        
        print(f"✅ Datos enviados a la base de datos simulada")
        print(f"{'🗄️'*20}\n")
        
        # Simular un pequeño delay como si fuera una operación real de BD
        await asyncio.sleep(0.1)
    
    def _validate_single_frame_message(self, data: Dict) -> bool:
        """Validar estructura del mensaje de frame único."""
        required_fields = ['image', 'metadata']
        if not all(field in data for field in required_fields):
            logger.error("Campos requeridos faltantes en el mensaje de frame único")
            return False
        
        if not isinstance(data['image'], str):
            logger.error("El campo 'image' debe ser una cadena base64")
            return False
        
        return True
    
    def _validate_frame_message(self, data: Dict) -> bool:
        """Validar estructura del mensaje de frames múltiples."""
        required_fields = ['frames', 'metadata']
        if not all(field in data for field in required_fields):
            logger.error("Campos requeridos faltantes en el mensaje")
            return False
        
        if not isinstance(data['frames'], list):
            logger.error("El campo 'frames' debe ser una lista")
            return False
        
        if len(data['frames']) == 0:
            logger.warning("Lista de frames vacía")
            return True  # Permitir listas vacías, retornar resultado vacío
        
        # Validar estructura de cada frame
        for i, frame in enumerate(data['frames']):
            if not isinstance(frame, dict):
                logger.error(f"Frame {i} debe ser un diccionario")
                return False
            
            if 'image' not in frame:
                logger.error(f"Frame {i} debe contener campo 'image'")
                return False
        
        return True
    
    async def _process_single_frame(self, data: Dict) -> Dict[str, Any]:
        """Procesar un solo frame para detección de objetos."""
        metadata = data['metadata']
        
        try:
            # Decodificar imagen base64
            image = self._decode_base64_image(data['image'])
            
            # Detectar objetos usando YOLO
            detections = self.yolo_detector.detect_objects(image)
            
            result = {
                'source': metadata.get('source', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown'),
                'detections': detections,
                'detection_count': len(detections)
            }
            
            logger.info(f"Frame procesado: {len(detections)} objetos detectados")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando frame único: {e}")
            return {
                'source': metadata.get('source', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown'),
                'error': str(e),
                'detections': [],
                'detection_count': 0
            }
    
    async def _process_frames(self, data: Dict) -> Dict[str, Any]:
        """Procesar lista de frames para detección de objetos."""
        frames = data['frames']
        metadata = data['metadata']
        video_name = metadata.get('video_name', 'unknown_video')
        
        results = {
            'video_name': video_name,
            'total_frames': len(frames),
            'processed_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'detections': [],
            'summary': {
                'total_objects_detected': 0,
                'frames_with_objects': 0,
                'average_objects_per_frame': 0.0
            }
        }
        
        if len(frames) == 0:
            logger.warning("No hay frames para procesar")
            return results
        
        logger.info(f"Procesando {len(frames)} frames del video: {video_name}")
        
        total_objects = 0
        frames_with_objects = 0
        
        for i, frame_data in enumerate(frames):
            try:
                # Decodificar imagen base64
                image = self._decode_base64_image(frame_data['image'])
                
                # Detectar objetos usando YOLO
                detections = self.yolo_detector.detect_objects(image)
                
                frame_result = {
                    'frame_index': i,
                    'timestamp': frame_data.get('timestamp', f'frame_{i}'),
                    'detections': detections,
                    'detection_count': len(detections),
                    'status': 'success'
                }
                
                results['detections'].append(frame_result)
                results['successful_frames'] += 1
                
                # Actualizar estadísticas
                if len(detections) > 0:
                    frames_with_objects += 1
                    total_objects += len(detections)
                
                logger.info(f"Frame {i} procesado: {len(detections)} objetos detectados")
                    
            except Exception as e:
                logger.error(f"Error procesando frame {i}: {e}")
                results['detections'].append({
                    'frame_index': i,
                    'timestamp': frame_data.get('timestamp', f'frame_{i}'),
                    'error': str(e),
                    'detections': [],
                    'detection_count': 0,
                    'status': 'error'
                })
                results['failed_frames'] += 1
        
        results['processed_frames'] = len(frames)
        
        # Calcular estadísticas finales
        results['summary']['total_objects_detected'] = total_objects
        results['summary']['frames_with_objects'] = frames_with_objects
        if results['successful_frames'] > 0:
            results['summary']['average_objects_per_frame'] = total_objects / results['successful_frames']
        
        return results
    
    def _decode_base64_image(self, base64_string: str) -> np.ndarray:
        """Decodificar imagen base64 a formato OpenCV con manejo robusto de errores."""
        try:
            if not base64_string:
                raise ValueError("Cadena base64 vacía")
            
            logger.info(f"Decodificando imagen base64 de {len(base64_string)} caracteres")
            
            # Limpiar el string base64
            clean_b64 = base64_string.strip()
            
            # Remover prefijo data URL si existe (data:image/jpeg;base64,)
            if ',' in clean_b64:
                clean_b64 = clean_b64.split(',')[1]
                logger.info("Prefijo data URL removido")
            
            # Estrategia 1: Intentar decodificación directa
            try:
                image_data = base64.b64decode(clean_b64)
                logger.info(f"✅ Decodificación directa exitosa: {len(image_data)} bytes")
                
            except Exception as e1:
                logger.warning(f"⚠️ Decodificación directa falló: {e1}")
                
                # Estrategia 2: Corregir padding y reintentar
                missing_padding = len(clean_b64) % 4
                if missing_padding:
                    clean_b64 += '=' * (4 - missing_padding)
                    logger.info(f"🔧 Padding corregido: agregados {4 - missing_padding} caracteres '='")
                
                try:
                    image_data = base64.b64decode(clean_b64)
                    logger.info(f"✅ Decodificación con padding exitosa: {len(image_data)} bytes")
                    
                except Exception as e2:
                    logger.error(f"❌ Decodificación con padding falló: {e2}")
                    
                    # Estrategia 3: Limpiar caracteres inválidos
                    valid_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
                    clean_b64 = ''.join(c for c in base64_string.strip() if c in valid_chars)
                    
                    # Corregir padding después de limpiar
                    missing_padding = len(clean_b64) % 4
                    if missing_padding:
                        clean_b64 += '=' * (4 - missing_padding)
                    
                    try:
                        image_data = base64.b64decode(clean_b64)
                        logger.info(f"✅ Decodificación con limpieza exitosa: {len(image_data)} bytes")
                    except Exception as e3:
                        logger.error(f"❌ Todas las estrategias de decodificación fallaron: {e1}, {e2}, {e3}")
                        raise ValueError(f"No se pudo decodificar base64: {e3}")
            
            # Convertir bytes a numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Decodificar imagen con OpenCV
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("cv2.imdecode retornó None - datos de imagen inválidos o formato no soportado")
            
            logger.info(f"✅ Imagen decodificada exitosamente: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"❌ Error general decodificando imagen base64: {e}")
            # Log información adicional para debugging
            logger.error(f"📊 Info de debugging - Base64 length: {len(base64_string)}, primeros 100 chars: {base64_string[:100]}")
            raise
    
    def _print_single_frame_result(self, result: Dict[str, Any]):
        """Imprimir resultado de detección de frame único."""
        print(f"\n{'='*50}")
        print(f"🖼️  DETECCIÓN EN FRAME ÚNICO")
        print(f"{'='*50}")
        print(f"Fuente: {result['source']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Objetos detectados: {result['detection_count']}")
        
        if result['detections']:
            for i, detection in enumerate(result['detections']):
                print(f"   {i+1}. {detection['class']}")
                print(f"      Confianza: {detection['confidence']:.2f}")
                print(f"      Posición: {detection['bbox']}")
        else:
            print("   ❌ No se detectaron objetos")
        
        print(f"{'='*50}\n")
    
    def _print_detection_results(self, results: Dict[str, Any]):
        """Imprimir resultados de detección múltiple."""
        print(f"\n{'='*70}")
        print(f"📹 RESULTADOS DE DETECCIÓN - {results['video_name']}")
        print(f"{'='*70}")
        print(f"Total de frames: {results['total_frames']}")
        print(f"Frames procesados exitosamente: {results['successful_frames']}")
        print(f"Frames con errores: {results['failed_frames']}")
        print(f"Total de objetos detectados: {results['summary']['total_objects_detected']}")
        print(f"Frames con objetos: {results['summary']['frames_with_objects']}")
        print(f"Promedio de objetos por frame: {results['summary']['average_objects_per_frame']:.2f}")
        
        # Mostrar detalles por frame (limitar a primeros 10 frames para no saturar la consola)
        frames_to_show = min(10, len(results['detections']))
        if frames_to_show < len(results['detections']):
            print(f"\nMostrando detalles de los primeros {frames_to_show} frames:")
        
        for i in range(frames_to_show):
            frame_result = results['detections'][i]
            frame_idx = frame_result['frame_index']
            timestamp = frame_result['timestamp']
            detections = frame_result.get('detections', [])
            status = frame_result.get('status', 'unknown')
            
            print(f"\n🎬 Frame {frame_idx} - {timestamp} [{status.upper()}]")
            
            if status == 'error':
                print(f"   ❌ Error: {frame_result.get('error', 'Error desconocido')}")
            else:
                print(f"   Objetos detectados: {len(detections)}")
                
                if detections:
                    for j, detection in enumerate(detections):
                        print(f"   {j+1}. {detection['class']}")
                        print(f"      Confianza: {detection['confidence']:.2f}")
                        print(f"      Posición: {detection['bbox']}")
                else:
                    print("   ❌ No se detectaron objetos")
        
        if len(results['detections']) > frames_to_show:
            remaining = len(results['detections']) - frames_to_show
            print(f"\n... y {remaining} frames adicionales procesados")
        
        print(f"{'='*70}\n")
    
    async def _send_simple_success_response(self, msg, message: str):
        """Enviar respuesta simple de éxito al NATS."""
        response = self.response_handler.create_simple_success_response(message)
        await msg.respond(json.dumps(response).encode())
        logger.info(f"Respuesta enviada al NATS: {message}")
    
    async def _send_simple_error_response(self, msg, error_message: str):
        """Enviar respuesta simple de error al NATS."""
        response = self.response_handler.create_simple_error_response(error_message)
        await msg.respond(json.dumps(response).encode())
        logger.error(f"Respuesta de error enviada al NATS: {error_message}")