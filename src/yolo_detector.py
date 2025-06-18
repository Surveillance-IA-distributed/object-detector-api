import cv2
import numpy as np
import os
import logging
from ultralytics import YOLO
from dotenv import load_dotenv
from typing import List, Dict, Any
import urllib.request

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)

class YOLOObjectDetector:
    """Detector de objetos usando YOLO."""

    def __init__(self):
        self.model_filename = os.getenv('YOLO_MODEL', 'yolov8n.pt')
        self.model_path = os.path.abspath(self.model_filename)
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
        self.model = None
        self._ensure_model_exists()
        self._load_model()

    def _ensure_model_exists(self):
        """Descargar el modelo si no está presente."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Modelo no encontrado en {self.model_path}. Descargando...")
            url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{self.model_filename}"
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




    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = {
                            'class': result.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': {
                                'x1': float(box.xyxy[0][0]),
                                'y1': float(box.xyxy[0][1]),
                                'x2': float(box.xyxy[0][2]),
                                'y2': float(box.xyxy[0][3])
                            }
                        }
                        detections.append(detection)

            logger.debug(f"Detectados {len(detections)} objetos")
            return detections

        except Exception as e:
            logger.error(f"Error en detección de objetos: {e}")
            return []

    def get_detection_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not detections:
            return {
                'total_objects': 0,
                'classes': {},
                'avg_confidence': 0.0
            }

        class_counts = {}
        confidences = []

        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(confidence)

        return {
            'total_objects': len(detections),
            'classes': class_counts,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'max_confidence': max(confidences) if confidences else 0.0,
            'min_confidence': min(confidences) if confidences else 0.0
        }
