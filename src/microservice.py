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
from .database_handler import DatabaseHandler 
from datetime import datetime
# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)



class ObjectDetectionMicroservice:  
    def __init__(self, nats_url: str = None, service_name: str = None):
        self.nats_url = nats_url or os.getenv('NATS_URL', 'nats://localhost:4222')
        self.service_name = service_name or os.getenv('SERVICE_NAME', 'object-detection-service')
        self.timeout = float(os.getenv('NATS_TIMEOUT', '30.0'))
        self.max_frames_per_batch = int(os.getenv('MAX_FRAMES_PER_BATCH', '100'))
        
        # Flag para reiniciar tracking entre lotes
        self.reset_tracking_between_batches = bool(os.getenv('RESET_TRACKING_BETWEEN_BATCHES', 'True'))
        
        self.nc = None
        self.yolo_detector = YOLOObjectDetector()  # Asumiendo que existe
        self.response_handler = ResponseHandler()  # Asumiendo que existe
        
        # Inicializar manejador de base de datos
        self.db_handler = DatabaseHandler()
        
    async def start(self):
        """Iniciar el microservicio y conectar a NATS y base de datos."""
        try:
            # Inicializar base de datos
            await self.db_handler.initialize()
            logger.info("Base de datos inicializada")
            
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
            await self.db_handler.close()
    
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
            
            # CORREGIDO: Enviar a base de datos real
            try:
                # Preparar datos para inserción individual
                if result.get('detections'):
                    detections_to_insert = []
                    metadata = data['metadata']
                    
                    # CORREGIDO: Manejar source como string o int
                    source_value = metadata.get('source', '1')
                    if isinstance(source_value, str) and not source_value.isdigit():
                        # Si source es un string no numérico, usar un ID por defecto o mapear
                        stream_session_id = 1  # Valor por defecto
                        logger.warning(f"Source '{source_value}' no es numérico, usando stream_session_id=1")
                    else:
                        stream_session_id = int(source_value)
                    
                    for detection in result['detections']:
                        detection_data = {
                            'stream_session_id': stream_session_id,  # CORREGIDO: usar stream_session_id seguro
                            'clase': detection['class'],
                            'x': detection['bbox']['x1'],
                            'y': detection['bbox']['y1'],
                            'ancho': detection['bbox']['x2'] - detection['bbox']['x1'],
                            'alto': detection['bbox']['y2'] - detection['bbox']['y1'],
                            'confianza': detection['confidence'],
                            'milisegundos': result['timestamp_ms'],
                            'detection_time': datetime.now().strftime('%H:%M:%S'),  # formato HH:MM:SS
                            'color': detection.get('color', {}).get('average'),
                            'color_superior': detection.get('color', {}).get('upper'),
                            'color_inferior': detection.get('color', {}).get('lower')
                        }
                        detections_to_insert.append(detection_data)
                    
                    # Insertar múltiples detecciones
                    detections_count = await self.db_handler.insert_multiple_detections(detections_to_insert)
                    logger.info(f"Frame único: {detections_count} detecciones guardadas en BD para stream_session_id {stream_session_id}")
                else:
                    detections_count = 0
                    logger.info("Frame único: sin detecciones para guardar")
                
                # También mostrar log detallado de lo que se guardó
                await self.log_processing_results(result, 'single_frame')
                
                # Enviar respuesta exitosa
                await self._send_simple_success_response(
                    msg, 
                    f"Frame procesado exitosamente: {detections_count} detecciones guardadas en BD"
                )
                
            except Exception as db_error:
                logger.error(f"Error guardando en base de datos: {db_error}")
                # Aún enviar respuesta pero indicar problema con BD
                await self._send_simple_success_response(
                    msg, 
                    f"Frame procesado pero error guardando en BD: {str(db_error)}"
                )
        
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
            
            # Reiniciar tracking si está configurado
            if self.reset_tracking_between_batches:
                self.yolo_detector.reset_tracking()
                logger.info("Tracking reiniciado para nuevo lote")
            
            # Procesar frames
            results = await self._process_frames(data)
            
            # Imprimir resultados de detección
            self._print_detection_results(results)
            
            # CORREGIDO: Enviar a base de datos real
            try:
                # Preparar todas las detecciones para inserción en lote
                all_detections = []
                metadata = data['metadata']
                
                # CORREGIDO: Manejar source como string o int
                source_value = metadata.get('source', '1')
                if isinstance(source_value, str) and not source_value.isdigit():
                    # Si source es un string no numérico, usar un ID por defecto o mapear
                    stream_session_id = 1  # Valor por defecto
                    logger.warning(f"Source '{source_value}' no es numérico, usando stream_session_id=1")
                else:
                    stream_session_id = int(source_value)
                
                for frame_result in results['detections']:
                    if frame_result.get('detections') and frame_result.get('status') == 'success':
                        for detection in frame_result['detections']:
                            detection_data = {
                                'stream_session_id': stream_session_id,  # CORREGIDO: usar stream_session_id seguro
                                'clase': detection['class'],
                                'x': detection['bbox']['x1'],
                                'y': detection['bbox']['y1'],
                                'ancho': detection['bbox']['x2'] - detection['bbox']['x1'],
                                'alto': detection['bbox']['y2'] - detection['bbox']['y1'],
                                'confianza': detection['confidence'],
                                'milisegundos': frame_result['timestamp_ms'],
                                'detection_time': datetime.now().strftime('%H:%M:%S'),  # formato HH:MM:SS
                                'color': detection.get('color', {}).get('average'),
                                'color_superior': detection.get('color', {}).get('upper'),
                                'color_inferior': detection.get('color', {}).get('lower')
                            }
                            all_detections.append(detection_data)
                
                # Insertar todas las detecciones en lote
                total_detections = await self.db_handler.insert_multiple_detections(all_detections)
                logger.info(f"Análisis múltiple: {total_detections} detecciones guardadas en BD para stream_session_id {stream_session_id}")
                
                # También mostrar log detallado de lo que se guardó
                await self.log_processing_results(results, 'multiple_frames')
                
                # Enviar respuesta exitosa
                await self._send_simple_success_response(
                    msg, 
                    f"Lote de {results['total_frames']} frames procesado: {total_detections} detecciones guardadas en BD"
                )
                
            except Exception as db_error:
                logger.error(f"Error guardando en base de datos: {db_error}")
                # Aún enviar respuesta pero indicar problema con BD
                await self._send_simple_success_response(
                    msg, 
                    f"Lote de {results['total_frames']} frames procesado pero error guardando en BD: {str(db_error)}"
                )
            
        except Exception as e:
            logger.error(f"Error procesando frames: {e}")
            await self._send_simple_error_response(msg, f"Error en procesamiento: {str(e)}")
        
    async def log_processing_results(self, results: Dict[str, Any], analysis_type: str):
        """Mostrar log detallado de los resultados procesados y guardados en BD."""
        print(f"\n{'🗄️'*20}")
        print(f"📊 RESULTADOS GUARDADOS EN BASE DE DATOS")
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
                    
                # Mostrar información de colores para personas
                persons = [d for d in results['detections'] if d['class'] == 'person']
                if persons:
                    print(f"  - Información de colores (personas):")
                    for i, person in enumerate(persons[:3]):  # Limitar a 3 personas
                        colors = person.get('color', {})
                        print(f"    Persona {i+1}:")
                        print(f"      Color promedio: {colors.get('average', 'N/A')}")
                        if colors.get('upper'):
                            print(f"      Color superior: {colors.get('upper', 'N/A')}")
                            print(f"      Color inferior: {colors.get('lower', 'N/A')}")
        
        elif analysis_type == 'multiple_frames':
            print(f"Datos del lote de frames:")
            print(f"  - Video: {results.get('video_name', 'N/A')}")
            print(f"  - Total frames: {results.get('total_frames', 0)}")
            print(f"  - Frames exitosos: {results.get('successful_frames', 0)}")
            print(f"  - Frames con error: {results.get('failed_frames', 0)}")
            print(f"  - Total objetos detectados: {results['summary']['total_objects_detected']}")
            print(f"  - Total personas detectadas: {results['summary'].get('total_persons_detected', 0)}")
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
        
        print(f"✅ Datos guardados exitosamente en PostgreSQL")
        print(f"{'🗄️'*20}\n")
    

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
    
    def _extract_timestamp_ms(self, timestamp_str: str) -> int:
        """Extraer timestamp en milisegundos desde string de timestamp."""
        try:
            # Si es un timestamp unix en string
            if timestamp_str.isdigit():
                return int(timestamp_str)
            
            # Si es formato "frame_X", extraer el número
            if timestamp_str.startswith('frame_'):
                frame_num = int(timestamp_str.split('_')[1])
                return frame_num * 33  # Asumir ~30 FPS (33ms por frame)
            
            # Si contiene timestamp en formato HH:MM:SS o similar
            if ':' in timestamp_str:
                parts = timestamp_str.split(':')
                if len(parts) >= 3:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = float(parts[2])
                    return int((hours * 3600 + minutes * 60 + seconds) * 1000)
            
            return 0
        except:
            return 0
    
    async def _process_single_frame(self, data: Dict) -> Dict[str, Any]:
        """Procesar un solo frame para detección de objetos."""
        metadata = data['metadata']
        
        try:
            # Decodificar imagen base64
            image = self._decode_base64_image(data['image'])
            
            # Extraer timestamp en milisegundos
            timestamp_str = metadata.get('timestamp', '0')
            current_time_ms = self._extract_timestamp_ms(timestamp_str)
            
            # Detectar objetos usando YOLO con análisis de colores
            detections = self.yolo_detector.detect_objects(image, current_time_ms)
            
            # Obtener resumen de detecciones
            summary = self.yolo_detector.get_detection_summary(detections)
            
            result = {
                'source': metadata.get('source', 'unknown'),
                'timestamp': timestamp_str,
                'timestamp_ms': current_time_ms,
                'detections': detections,
                'detection_count': len(detections),
                'summary': summary
            }
            
            logger.info(f"Frame procesado: {len(detections)} objetos detectados ({summary.get('total_persons', 0)} personas)")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando frame único: {e}")
            return {
                'source': metadata.get('source', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown'),
                'timestamp_ms': 0,
                'error': str(e),
                'detections': [],
                'detection_count': 0,
                'summary': {}
            }
    
    async def _process_frames(self, data: Dict) -> Dict[str, Any]:
        """Procesar lista de frames para detección de objetos."""
        frames = data['frames']
        metadata = data['metadata']
        video_name = metadata.get('video_name', 'unknown_video')
        source = metadata.get('source', 'unknown_source')  # AÑADIDO: Obtener source
        
        results = {
            'video_name': video_name,
            'source': source,  # AÑADIDO: Incluir source en los resultados
            'total_frames': len(frames),
            'processed_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'detections': [],
            'summary': {
                'total_objects_detected': 0,
                'total_persons_detected': 0,
                'frames_with_objects': 0,
                'average_objects_per_frame': 0.0,
                'unique_objects_tracked': 0
            }
        }
        
        if len(frames) == 0:
            logger.warning("No hay frames para procesar")
            return results
        
        logger.info(f"Procesando {len(frames)} frames del video: {video_name} (source: {source})")
        
        total_objects = 0
        total_persons = 0
        frames_with_objects = 0
        
        for i, frame_data in enumerate(frames):
            try:
                # Decodificar imagen base64
                image = self._decode_base64_image(frame_data['image'])
                
                # Extraer timestamp en milisegundos
                timestamp_str = frame_data.get('timestamp', f'frame_{i}')
                current_time_ms = self._extract_timestamp_ms(timestamp_str)
                
                # Detectar objetos usando YOLO con análisis de colores
                detections = self.yolo_detector.detect_objects(image, current_time_ms)
                
                # Obtener resumen de detecciones para este frame
                frame_summary = self.yolo_detector.get_detection_summary(detections)
                
                frame_result = {
                    'frame_index': i,
                    'timestamp': timestamp_str,
                    'timestamp_ms': current_time_ms,
                    'detections': detections,
                    'detection_count': len(detections),
                    'summary': frame_summary,
                    'status': 'success'
                }
                
                results['detections'].append(frame_result)
                results['successful_frames'] += 1
                
                # Actualizar estadísticas globales
                if len(detections) > 0:
                    frames_with_objects += 1
                    total_objects += len(detections)
                    total_persons += frame_summary.get('total_persons', 0)
                
                logger.info(f"Frame {i} procesado: {len(detections)} objetos detectados ({frame_summary.get('total_persons', 0)} personas)")
                    
            except Exception as e:
                logger.error(f"Error procesando frame {i}: {e}")
                results['detections'].append({
                    'frame_index': i,
                    'timestamp': frame_data.get('timestamp', f'frame_{i}'),
                    'timestamp_ms': 0,
                    'error': str(e),
                    'detections': [],
                    'detection_count': 0,
                    'summary': {},
                    'status': 'error'
                })
                results['failed_frames'] += 1
        
        results['processed_frames'] = len(frames)
        
        # Calcular estadísticas finales
        results['summary']['total_objects_detected'] = total_objects
        results['summary']['total_persons_detected'] = total_persons
        results['summary']['frames_with_objects'] = frames_with_objects
        results['summary']['unique_objects_tracked'] = len(self.yolo_detector.detected_objects)
        
        if results['successful_frames'] > 0:
            results['summary']['average_objects_per_frame'] = total_objects / results['successful_frames']
        
        return results
    
    def _decode_base64_image(self, base64_string: str, save_for_debug: bool = True) -> np.ndarray:
        """Decodificar imagen base64 a formato OpenCV con manejo robusto de errores y guardado opcional."""
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
            
            # NUEVO: Guardar datos binarios para debugging si está habilitado
            if save_for_debug:
                try:
                    # Crear directorio de debugging si no existe
                    debug_dir = os.getenv('DEBUG_IMAGES_DIR', './debug_images')
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)
                        logger.info(f"Directorio de debugging creado: {debug_dir}")
                    
                    # Generar nombre único para la imagen
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Microsegundos truncados
                    raw_filename = f"raw_image_{timestamp}.bin"
                    raw_path = os.path.join(debug_dir, raw_filename)
                    
                    # Guardar datos binarios
                    with open(raw_path, 'wb') as f:
                        f.write(image_data)
                    logger.info(f"💾 Datos binarios guardados: {raw_path} ({len(image_data)} bytes)")
                    
                except Exception as save_error:
                    logger.warning(f"⚠️ No se pudieron guardar datos binarios: {save_error}")
            
            # Convertir bytes a numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Decodificar imagen con OpenCV
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                # NUEVO: Intentar diferentes formatos de decodificación
                logger.warning("cv2.imdecode con IMREAD_COLOR falló, probando otros formatos...")
                
                # Intentar con IMREAD_UNCHANGED
                image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    logger.info("✅ Decodificación exitosa con IMREAD_UNCHANGED")
                    # Convertir a BGR si es necesario
                    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                        logger.info("Imagen convertida de RGBA a BGR")
                    elif len(image.shape) == 2:  # Grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        logger.info("Imagen convertida de escala de grises a BGR")
                else:
                    # Intentar con IMREAD_ANYDEPTH | IMREAD_ANYCOLOR
                    image = cv2.imdecode(nparr, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                    if image is not None:
                        logger.info("✅ Decodificación exitosa con IMREAD_ANYDEPTH | IMREAD_ANYCOLOR")
                    else:
                        raise ValueError("cv2.imdecode retornó None - datos de imagen inválidos o formato no soportado")
            
            # NUEVO: Verificar calidad y características de la imagen
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            
            logger.info(f"✅ Imagen decodificada exitosamente:")
            logger.info(f"   📏 Dimensiones: {width}x{height} píxeles")
            logger.info(f"   🎨 Canales: {channels}")
            logger.info(f"   📊 Tipo de datos: {image.dtype}")
            logger.info(f"   💾 Tamaño en memoria: {image.nbytes} bytes")
            
            # Calcular estadísticas básicas de calidad
            if channels >= 3:
                mean_brightness = np.mean(image)
                std_brightness = np.std(image)
                logger.info(f"   💡 Brillo promedio: {mean_brightness:.2f}")
                logger.info(f"   📈 Desviación estándar: {std_brightness:.2f}")
                
                # Detectar si la imagen está muy oscura o muy clara
                if mean_brightness < 30:
                    logger.warning("⚠️ Imagen muy oscura (brillo < 30)")
                elif mean_brightness > 225:
                    logger.warning("⚠️ Imagen muy clara (brillo > 225)")
                
                if std_brightness < 10:
                    logger.warning("⚠️ Imagen con poco contraste (std < 10)")
            
            # NUEVO: Guardar imagen decodificada para verificación visual
            if save_for_debug:
                try:
                    decoded_filename = f"decoded_image_{timestamp}.jpg"
                    decoded_path = os.path.join(debug_dir, decoded_filename)
                    
                    # Guardar con alta calidad
                    cv2.imwrite(decoded_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    logger.info(f"🖼️ Imagen decodificada guardada: {decoded_path}")
                    
                    # También guardar una versión PNG sin pérdida si la imagen es pequeña
                    if width * height < 1000000:  # Menos de 1MP
                        png_filename = f"decoded_image_{timestamp}.png"
                        png_path = os.path.join(debug_dir, png_filename)
                        cv2.imwrite(png_path, image)
                        logger.info(f"🖼️ Versión PNG guardada: {png_path}")
                    
                except Exception as save_error:
                    logger.warning(f"⚠️ No se pudo guardar imagen decodificada: {save_error}")
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Error general decodificando imagen base64: {e}")
            # Log información adicional para debugging
            logger.error(f"📊 Info de debugging - Base64 length: {len(base64_string)}")
            logger.error(f"📊 Primeros 100 chars: {base64_string[:100]}")
            logger.error(f"📊 Últimos 100 chars: {base64_string[-100:]}")
            
            # Analizar el formato del base64
            if base64_string.startswith('data:'):
                header = base64_string.split(',')[0]
                logger.error(f"📊 Header data URL: {header}")
            
            raise
    
    def _print_single_frame_result(self, result: Dict[str, Any]):
        """Imprimir resultado de detección de frame único."""
        print(f"\n{'='*60}")
        print(f"🖼️  DETECCIÓN EN FRAME ÚNICO")
        print(f"{'='*60}")
        print(f"Fuente: {result['source']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Objetos detectados: {result['detection_count']}")
        
        # Mostrar resumen si existe
        summary = result.get('summary', {})
        if summary:
            print(f"Personas detectadas: {summary.get('total_persons', 0)}")
            print(f"Confianza promedio: {summary.get('avg_confidence', 0):.2f}")
        
        if result['detections']:
            print(f"\n📋 DETECCIONES DETALLADAS:")
            for i, detection in enumerate(result['detections']):
                print(f"   {i+1}. {detection['class']} (ID: {detection.get('id', 'N/A')})")
                print(f"      Confianza: {detection['confidence']:.2f}")
                print(f"      Posición: x1={detection['bbox']['x1']:.1f}, y1={detection['bbox']['y1']:.1f}")
                print(f"                x2={detection['bbox']['x2']:.1f}, y2={detection['bbox']['y2']:.1f}")
                
                # Mostrar información de colores
                colors = detection.get('color', {})
                if colors.get('average'):
                    print(f"      Color promedio: {colors['average']}")
                    if colors.get('upper') and colors.get('lower'):
                        print(f"      Color superior: {colors['upper']}")
                        print(f"      Color inferior: {colors['lower']}")
        else:
            print("   ❌ No se detectaron objetos")
        
        print(f"{'='*60}\n")
    
    def _print_detection_results(self, results: Dict[str, Any]):
        """Imprimir resultados de detección múltiple."""
        print(f"\n{'='*80}")
        print(f"📹 RESULTADOS DE DETECCIÓN - {results['video_name']}")
        print(f"{'='*80}")
        print(f"Total de frames: {results['total_frames']}")
        print(f"Frames procesados exitosamente: {results['successful_frames']}")
        print(f"Frames con errores: {results['failed_frames']}")
        print(f"Total de objetos detectados: {results['summary']['total_objects_detected']}")
        print(f"Total de personas detectadas: {results['summary']['total_persons_detected']}")
        print(f"Objetos únicos rastreados: {results['summary']['unique_objects_tracked']}")
        print(f"Frames con objetos: {results['summary']['frames_with_objects']}")
        print(f"Promedio de objetos por frame: {results['summary']['average_objects_per_frame']:.2f}")
        
        # Mostrar detalles por frame (limitar a primeros 5 frames para no saturar la consola)
        frames_to_show = min(5, len(results['detections']))
        if frames_to_show < len(results['detections']):
            print(f"\nMostrando detalles de los primeros {frames_to_show} frames:")
        
        for i in range(frames_to_show):
            frame_result = results['detections'][i]
            frame_idx = frame_result['frame_index']
            timestamp = frame_result['timestamp']
            detections = frame_result.get('detections', [])
            status = frame_result.get('status', 'unknown')
            frame_summary = frame_result.get('summary', {})
            
            print(f"\n🎬 Frame {frame_idx} - {timestamp} [{status.upper()}]")
            
            if status == 'error':
                print(f"   ❌ Error: {frame_result.get('error', 'Error desconocido')}")
            else:
                print(f"   Objetos detectados: {len(detections)} ({frame_summary.get('total_persons', 0)} personas)")
                
                if detections:
                    # Mostrar solo las primeras 3 detecciones para no saturar
                    detections_to_show = min(3, len(detections))
                    for j in range(detections_to_show):
                        detection = detections[j]
                        print(f"   {j+1}. {detection['class']} (ID: {detection.get('id', 'N/A')})")
                        print(f"      Confianza: {detection['confidence']:.2f}")
                        
                        # Mostrar color para personas
                        if detection['class'] == 'person':
                            colors = detection.get('color', {})
                            if colors.get('upper') and colors.get('lower'):
                                print(f"      Colores: {colors['upper']} / {colors['lower']}")
                    
                    if len(detections) > detections_to_show:
                        print(f"   ... y {len(detections) - detections_to_show} detecciones más")
                else:
                    print("   ❌ No se detectaron objetos")
        
        if len(results['detections']) > frames_to_show:
            remaining = len(results['detections']) - frames_to_show
            print(f"\n... y {remaining} frames adicionales procesados")
        
        print(f"{'='*80}\n")
    
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