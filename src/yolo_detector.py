#src/yolo_detector
import cv2
import numpy as np
import os
import logging
from ultralytics import YOLO
from dotenv import load_dotenv
from typing import List, Dict, Any
import urllib.request
import webcolors
import torch
from datetime import timedelta

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)

class YOLOObjectDetector:
    """Detector de objetos usando YOLO con análisis de colores."""

    def __init__(self):
        # Configuración del modelo - actualizado para usar YOLO11
        self.model_filename = os.getenv('YOLO_MODEL', 'yolo11n.pt')
        self.model_path = os.path.abspath(self.model_filename)
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
        
        # Configuración para seguimiento de objetos
        self.distance_threshold = int(os.getenv('DISTANCE_THRESHOLD', '30'))
        self.detected_objects = []
        
        self.model = None
        self._ensure_model_exists()
        self._load_model()

    def _ensure_model_exists(self):
        """Descargar el modelo si no está presente."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Modelo no encontrado en {self.model_path}. Descargando...")
            # URL actualizada para YOLO11
            url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{self.model_filename}"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                logger.info(f"Modelo descargado exitosamente en: {self.model_path}")
            except Exception as e:
                logger.error(f"Fallo al descargar el modelo: {e}")
                raise

    def _load_model(self):
        """Cargar el modelo YOLO directamente desde el archivo .pt usando Ultralytics."""
        try:
            logger.info(f"Cargando modelo YOLO desde: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Modelo YOLO cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo YOLO: {e}")
            raise

    def euclidean_distance(self, p1: tuple, p2: tuple) -> float:
        """Calcular distancia euclidiana entre dos puntos."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_average_color(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Obtener color promedio de una región del frame."""
        try:
            # Asegurar que las coordenadas estén dentro de los límites del frame
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.array([0, 0, 0])  # Color negro por defecto
            
            object_region = frame[y1:y2, x1:x2]
            if object_region.size == 0:
                return np.array([0, 0, 0])
            
            object_region_rgb = cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB)
            avg_color = np.mean(object_region_rgb, axis=(0, 1))
            return avg_color
        except Exception as e:
            logger.error(f"Error calculando color promedio: {e}")
            return np.array([0, 0, 0])

    def get_upper_lower_colors(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple:
        """Obtener colores de la parte superior e inferior de una persona."""
        try:
            upper_y2 = int((y1 + y2) / 2)
            upper_color = self.get_average_color(frame, x1, y1, x2, upper_y2)
            lower_y1 = upper_y2
            lower_color = self.get_average_color(frame, x1, lower_y1, x2, y2)
            return upper_color, lower_color
        except Exception as e:
            logger.error(f"Error calculando colores superior/inferior: {e}")
            return np.array([0, 0, 0]), np.array([0, 0, 0])

    def rgb_to_name(self, rgb: tuple) -> str:
        """Convertir RGB a nombre de color."""
        try:
            rgb = tuple(int(round(c)) for c in rgb)
            color_name = webcolors.rgb_to_name(rgb)
            return color_name
        except ValueError:
            try:
                hex_value = webcolors.rgb_to_hex(rgb)
                return hex_value
            except Exception:
                return "#000000"  # Negro por defecto

    def detect_objects(self, image: np.ndarray, current_time: int = 0) -> List[Dict[str, Any]]:
        """
        Detectar objetos en una imagen con análisis de colores.
        
        Args:
            image: Imagen de entrada
            current_time: Tiempo actual en milisegundos (opcional)
        
        Returns:
            Lista de detecciones con información extendida
        """
        try:
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            detections = []
            person_count = 0

            # Calcular timestamp
            timestamp = str(timedelta(milliseconds=current_time)) if current_time > 0 else "0:00:00"

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = result.boxes.xyxy
                confidences = result.boxes.conf
                class_ids = result.boxes.cls
                class_names = result.names

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    w = x2 - x1
                    h = y2 - y1
                    class_name = class_names[int(class_ids[i])]
                    confidence = float(confidences[i]) if confidences is not None else 0.0

                    # Convertir coordenadas a tipos nativos de Python
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    w, h = float(w), float(h)

                    # Centro del objeto
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    # Verificar si el objeto ya fue detectado (evitar duplicados)
                    object_id = None
                    for obj in self.detected_objects:
                        prev_cx, prev_cy = obj['center']
                        if self.euclidean_distance((cx, cy), (prev_cx, prev_cy)) < self.distance_threshold:
                            object_id = obj['id']
                            break

                    # Si es un objeto nuevo, procesarlo
                    if object_id is None:
                        object_id = len(self.detected_objects) + 1
                        self.detected_objects.append({'id': object_id, 'center': (cx, cy)})

                        # Obtener color promedio
                        avg_color = self.get_average_color(image, int(x1), int(y1), int(x2), int(y2))
                        avg_color_name = self.rgb_to_name(tuple(avg_color))

                        detection = {
                            'id': object_id,
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': {
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'width': w,
                                'height': h
                            },
                            'center': {
                                'x': cx,
                                'y': cy
                            },
                            'timestamp': {
                                'milliseconds': float(current_time),
                                'formatted': timestamp
                            },
                            'color': {
                                'average': avg_color_name,
                                'upper': None,
                                'lower': None
                            }
                        }

                        # Si es una persona, obtener colores superior e inferior
                        if class_name == "person":
                            person_count += 1
                            upper_color, lower_color = self.get_upper_lower_colors(
                                image, int(x1), int(y1), int(x2), int(y2)
                            )
                            upper_color_name = self.rgb_to_name(tuple(upper_color))
                            lower_color_name = self.rgb_to_name(tuple(lower_color))
                            
                            detection['color']['upper'] = upper_color_name
                            detection['color']['lower'] = lower_color_name

                        detections.append(detection)

            logger.debug(f"Detectados {len(detections)} objetos nuevos ({person_count} personas)")
            return detections

        except Exception as e:
            logger.error(f"Error en detección de objetos: {e}")
            return []

    def get_detection_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Obtener resumen de las detecciones."""
        if not detections:
            return {
                'total_objects': 0,
                'total_persons': 0,
                'classes': {},
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0
            }

        class_counts = {}
        confidences = []
        person_count = 0

        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(confidence)
            
            if class_name == "person":
                person_count += 1

        return {
            'total_objects': len(detections),
            'total_persons': person_count,
            'classes': class_counts,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'max_confidence': max(confidences) if confidences else 0.0,
            'min_confidence': min(confidences) if confidences else 0.0
        }

    def reset_tracking(self):
        """Reiniciar el seguimiento de objetos."""
        self.detected_objects = []
        logger.info("Seguimiento de objetos reiniciado")

    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Dibujar las detecciones en la imagen."""
        image_copy = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # Dibujar rectángulo
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar etiqueta
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return image_copy