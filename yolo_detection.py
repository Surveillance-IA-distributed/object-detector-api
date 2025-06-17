from ultralytics import YOLO
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import webcolors  
from datetime import timedelta


# Cargar el modelo YOLOv8 preentrenado
# model = YOLO("yolov8n.pt")  
model = YOLO("yolo11n.pt")

# Ruta del video
video_path = "VIRAT_S_010200_08_000838_000867.mp4"

# Abrir el video
cap = cv2.VideoCapture(video_path)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Configuración de la velocidad (número de cuadros a saltar -3 )
speed_factor = 3  
frame_count = 0

# Seguimiento de los objetos 
detected_objects = []  

# Umbral de distancia evitar el mismo objeto
distance_threshold = 30  

# Conteo de personas por fotograma
person_count_per_frame = []


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# color de objetos
def get_average_color(frame, x1, y1, x2, y2):
    # Extraer la región del objeto utilizando las coordenadas de la caja
    object_region = frame[y1:y2, x1:x2]
    object_region_rgb = cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB)
    avg_color = np.mean(object_region_rgb, axis=(0, 1))  # Promedio en los ejes (alto, ancho)
    return avg_color

# color de personas (upper - lower)
def get_upper_lower_colors(frame, x1, y1, x2, y2):
    upper_y2 = int((y1 + y2) / 2)  
    upper_color = get_average_color(frame, int(x1), int(y1), int(x2), upper_y2)
    lower_y1 = upper_y2
    lower_color = get_average_color(frame, int(x1), int(lower_y1), int(x2), int(y2))
    return upper_color, lower_color

# Función para convertir RGB a nombre de color, usar mejor un diccionario de colores
def rgb_to_name(rgb):
    try:
        rgb = tuple(int(round(c)) for c in rgb)  # Convertir entero        
        color_name = webcolors.rgb_to_name(rgb) # color más cercano
        return color_name

    except ValueError: # Sin nombre exacto, devolver el color hexadecimal
        hex_value = webcolors.rgb_to_hex(rgb)
        return hex_value


# Crear una lista para almacenar todas las detecciones
detections = []

try: #cada fotograma
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        if frame_count % speed_factor == 0:
            current_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))  # Obtener tiempo en milisegundos
            timestamp = str(timedelta(milliseconds=current_time)) 
            results = model(frame)
            person_count = 0  # Contador de personas en este fotograma

            for result in results: 
                if result.boxes is None or len(result.boxes) == 0:
                    print("No se encontraron detecciones en este fotograma.")
                    continue

                boxes = result.boxes.xyxy  # Coordenadas de las cajas (x1, y1, x2, y2)
                confidences = result.probs  # Confianza de las predicciones
                class_ids = result.boxes.cls  # Clases predichas
                class_names = result.names  # Diccionario con los nombres de las clases
                # es necesario poner confianza???????? creo q no 
                # Iterar sobre las detecciones y agregar los resultados a la lista de detecciones
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]  
                    w = x2 - x1
                    h = y2 - y1
                    class_name = class_names[int(class_ids[i])]  # Nombre de la clase
                    confidence = confidences[i] if confidences is not None else 0.0  # Confianza de la detección

                    # centro objeto
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    # convertir a numerico torchito -> float
                    cx = float(cx.item()) if isinstance(cx, torch.Tensor) else float(cx)
                    cy = float(cy.item()) if isinstance(cy, torch.Tensor) else float(cy)

                    #Si el centro del objeto ya ha sido detectado, no guardar (umbral)
                    object_id = None
                    for obj in detected_objects:
                        prev_cx, prev_cy = obj['center']
                        if euclidean_distance((cx, cy), (prev_cx, prev_cy)) < distance_threshold:
                            object_id = obj['id']  # Si están cerca, consideramos el mismo objeto
                            break

                    # guardar  JSON
                    if object_id is None:
                        # Generar un ID único para el objeto
                        object_id = len(detected_objects) + 1  # ID único basado en el tamaño de la lista
                        detected_objects.append({'id': object_id, 'center': (cx, cy)})  # Registrar el objeto
                        avg_color = get_average_color(frame, int(x1), int(y1), int(x2), int(y2))
                        avg_color_name = rgb_to_name(tuple(avg_color))

                        # Si el objeto es de la clase "persona", obtener los colores de la parte superior e inferior
                        if class_name == "person":
                            person_count += 1  # Incrementar el conteo de personas
                            upper_color, lower_color = get_upper_lower_colors(frame, int(x1), int(y1), int(x2), int(y2))
                            upper_color_name = rgb_to_name(tuple(upper_color))
                            lower_color_name = rgb_to_name(tuple(lower_color))

                            # JSON
                            detections.append({
                                'Clase': class_name,
                                'X': float(x1),
                                'Y': float(y1),
                                'Ancho': float(w),
                                'Alto': float(h),
                                'Confianza': float(confidence),
                                'Milisegundos': float(current_time),
                                'Timestamp': timestamp,
                                'Color': avg_color_name,
                                'ColorSuperior': upper_color_name,
                                'ColorInferior': lower_color_name
                            })
                        else:
                            # Para otros objetos, solo se guarda el color promedio
                            detections.append({
                                'Clase': class_name,
                                'X': float(x1),
                                'Y': float(y1),
                                'Ancho': float(w),
                                'Alto': float(h),
                                'Confianza': float(confidence),
                                'Milisegundos': float(current_time),
                                'Timestamp': timestamp,
                                'Color': avg_color_name,
                                'ColorSuperior': None,
                                'ColorInferior': None
                            })

                        # Mostrar en video nombre clase y el box
                        label = class_name
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  

            # Conteo
            person_count_per_frame.append(person_count)

            # Anotar el fotograma con los resultados
            scale_frame = 0.6
            frame = cv2.resize(frame, (0, 0), fx=scale_frame, fy=scale_frame)
            
            # Descomentar para TESTING
            # cv2.imshow('frame', frame)

        frame_count += 1

        # Salir si se presiona la tecla 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Guardar en archivo 
    with open('detections.json', 'w') as json_file:
        json.dump(detections, json_file, indent=4)

    # plot histo de contero por fotograma
    plt.plot(person_count_per_frame)
    plt.xlabel('Número de fotograma')
    plt.ylabel('Número de personas detectadas')
    plt.title('Conteo de personas por fotograma')
    plt.show()

    cap.release()

    # Descomentar para TESTING
    # cv2.destroyAllWindows()
