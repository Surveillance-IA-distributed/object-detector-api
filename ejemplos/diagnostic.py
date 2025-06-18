import base64
import cv2
import json
import re

def diagnose_base64_issue(image_path):
    """
    Diagnostica problemas con la codificación base64
    """
    print("🔍 DIAGNÓSTICO DE BASE64")
    print("=" * 50)
    
    try:
        # 1. Cargar imagen original
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ No se pudo cargar la imagen: {image_path}")
            return
        
        print(f"✅ Imagen cargada: {image.shape}")
        
        # 2. Codificar imagen
        _, buffer = cv2.imencode('.jpg', image)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        print(f"📊 Base64 generado:")
        print(f"   • Longitud: {len(base64_str)} caracteres")
        print(f"   • Padding válido: {len(base64_str) % 4 == 0}")
        print(f"   • Primeros 50 chars: {base64_str[:50]}...")
        print(f"   • Últimos 50 chars: ...{base64_str[-50:]}")
        
        # 3. Verificar caracteres válidos
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        invalid_chars = set(base64_str) - valid_chars
        
        if invalid_chars:
            print(f"⚠️  Caracteres inválidos encontrados: {invalid_chars}")
        else:
            print("✅ Todos los caracteres son válidos")
        
        # 4. Simular el mensaje que se envía
        message = {
            "frames": [
                {
                    "image": base64_str,
                    "timestamp": "2025-06-16T01:11:03.234752"
                }
            ],
            "metadata": {
                "video_name": "test_diagnostic",
                "source": "diagnostic",
                "processing_date": "2025-06-16"
            }
        }
        
        # 5. Serializar a JSON
        json_str = json.dumps(message)
        print(f"📨 Mensaje JSON:")
        print(f"   • Tamaño total: {len(json_str)} caracteres")
        
        # 6. Intentar deserializar y extraer base64
        parsed_message = json.loads(json_str)
        extracted_base64 = parsed_message['frames'][0]['image']
        
        print(f"📥 Base64 extraído del JSON:")
        print(f"   • Longitud: {len(extracted_base64)} caracteres")
        print(f"   • Coincide con original: {extracted_base64 == base64_str}")
        
        # 7. Intentar decodificar
        try:
            decoded_data = base64.b64decode(extracted_base64)
            print(f"✅ Decodificación exitosa: {len(decoded_data)} bytes")
            
            # 8. Intentar reconstruir imagen
            import numpy as np
            nparr = np.frombuffer(decoded_data, np.uint8)
            decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if decoded_image is not None:
                print(f"✅ Imagen reconstruida: {decoded_image.shape}")
                # Guardar imagen de prueba
                cv2.imwrite("diagnostic_output.jpg", decoded_image)
                print("💾 Imagen guardada como 'diagnostic_output.jpg'")
            else:
                print("❌ Error reconstruyendo imagen con cv2.imdecode")
                
        except Exception as e:
            print(f"❌ Error decodificando base64: {e}")
            
            # Intentar corregir padding
            print("🔧 Intentando corregir padding...")
            missing_padding = len(extracted_base64) % 4
            if missing_padding:
                corrected_base64 = extracted_base64 + '=' * (4 - missing_padding)
                print(f"   • Padding agregado: {4 - missing_padding} caracteres '='")
                
                try:
                    decoded_data = base64.b64decode(corrected_base64)
                    print(f"✅ Decodificación con padding corregido exitosa: {len(decoded_data)} bytes")
                except Exception as e2:
                    print(f"❌ Aún falla con padding corregido: {e2}")
            else:
                print("   • Padding ya es correcto")
        
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")

def test_microservice_base64_handling():
    """
    Simula el manejo de base64 del microservicio
    """
    print("\n🔬 SIMULACIÓN DEL MICROSERVICIO")
    print("=" * 50)
    
    # Simular recepción del mensaje
    test_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="  # 1x1 pixel PNG
    
    print(f"📨 Base64 de prueba recibido: {len(test_base64)} caracteres")
    
    # Simular el proceso del microservicio
    try:
        # Paso 1: Limpiar
        cleaned = test_base64.strip()
        print(f"🧹 Después de limpiar: {len(cleaned)} caracteres")
        
        # Paso 2: Verificar padding
        missing_padding = len(cleaned) % 4
        if missing_padding:
            cleaned += '=' * (4 - missing_padding)
            print(f"🔧 Padding corregido: +{4 - missing_padding} caracteres")
        
        # Paso 3: Decodificar
        decoded = base64.b64decode(cleaned)
        print(f"✅ Decodificado: {len(decoded)} bytes")
        
        # Paso 4: Convertir a imagen
        import numpy as np
        nparr = np.frombuffer(decoded, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is not None:
            print(f"✅ Imagen creada: {image.shape}")
        else:
            print("❌ cv2.imdecode falló")
            
    except Exception as e:
        print(f"❌ Error en simulación: {e}")

if __name__ == "__main__":
    # Ejecutar diagnóstico
    image_path = "imagenes/test_1.jpg"  # Ajusta la ruta
    diagnose_base64_issue(image_path)
    test_microservice_base64_handling()