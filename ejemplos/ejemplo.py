# ejemplos/ejemplo_uso.py
import asyncio
import json
import base64
import nats
import cv2
from datetime import datetime

async def ejemplo_analisis_con_imagen_real():
    """Ejemplo usando una imagen real desde disco."""
    print("📸 Iniciando ejemplo con imagen real...")
    
    # Conectar a NATS
    nc = await nats.connect("nats://localhost:4222")
    
    try:
        # Cargar una imagen real desde el sistema de archivos
        ruta_imagen = "imagenes/personas.jpg"  # Cambia esta ruta según tu entorno
        img = cv2.imread(ruta_imagen)
        
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen en: {ruta_imagen}")
        
        # Mostrar información de la imagen cargada
        height, width, channels = img.shape
        print(f"🖼️  Imagen cargada exitosamente:")
        print(f"   • Ruta: {ruta_imagen}")
        print(f"   • Dimensiones: {width}x{height}")
        print(f"   • Canales: {channels}")
        
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"   • Tamaño base64: {len(img_base64)} caracteres")
        
        # Crear mensaje con imagen real
        mensaje = {
            "frames": [
                {
                    "image": img_base64,
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "metadata": {
                "video_name": "imagen_real_test",
                "source": "video_demo",
                "processing_date": datetime.now().strftime("%Y-%m-%d")
            }
        }
        
        print("\n📤 Enviando imagen real para análisis...")
        print(f"📊 Detalles del mensaje enviado:")
        print(f"   • Total frames: {len(mensaje['frames'])}")
        print(f"   • Video name: {mensaje['metadata']['video_name']}")
        print(f"   • Source: {mensaje['metadata']['source']}")
        
        # Mostrar estructura del mensaje (sin la imagen base64 completa)
        mensaje_preview = mensaje.copy()
        mensaje_preview['frames'][0]['image'] = f"[BASE64_IMAGE_{len(img_base64)}_CHARS]"
        print(f"\n🔍 Estructura del mensaje enviado:")
        print(json.dumps(mensaje_preview, indent=2, ensure_ascii=False))
        
        print(f"\n⏳ Esperando respuesta del microservicio...")
        
        response = await nc.request(
            "object_detection.analyze_frames",
            json.dumps(mensaje).encode(),
            timeout=30.0
        )
        
        # Mostrar la respuesta pura/cruda
        respuesta_cruda = response.data.decode()
        print(f"\n📥 RESPUESTA CRUDA DEL MICROSERVICIO:")
        print(f"{'='*60}")
        print(respuesta_cruda)
        print(f"{'='*60}")
        
        # Parsear la respuesta
        resultado = json.loads(respuesta_cruda)
        
        print(f"\n📋 RESPUESTA PARSEADA:")
        print(f"{'='*60}")
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
        print(f"{'='*60}")
        
        # Procesar el resultado
        if resultado['success']:
            print(f"\n✅ ¡Análisis con imagen real exitoso!")
            print(f"🕐 Timestamp de respuesta: {resultado['timestamp']}")
            print(f"📨 Mensaje: {resultado['message']}")
            print(f"📊 Estado: {resultado['status']}")
            
            # Nota: Con la nueva implementación, los detalles están en los logs del servidor,
            # no en la respuesta NATS
            print(f"\n💡 Nota: Los detalles de detección se procesan en el servidor")
            print(f"   y se envían a la base de datos. La respuesta NATS es simplificada.")
            
        else:
            print(f"\n❌ Error en el análisis:")
            print(f"🕐 Timestamp: {resultado['timestamp']}")
            print(f"📨 Mensaje de error: {resultado['message']}")
            print(f"📊 Estado: {resultado['status']}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: No se encontró la imagen - {e}")
        print("💡 Sugerencias:")
        print("   • Verifica que la ruta de la imagen sea correcta")
        print("   • Asegúrate de que el archivo existe")
        print("   • Prueba con una ruta absoluta")
        
    except nats.errors.TimeoutError:
        print(f"\n⏰ Timeout: El microservicio no respondió en 30 segundos")
        print("💡 Sugerencias:")
        print("   • Verifica que el microservicio esté corriendo")
        print("   • Revisa la conexión a NATS")
        print("   • Considera aumentar el timeout")
        
    except json.JSONDecodeError as e:
        print(f"\n❌ Error decodificando respuesta JSON: {e}")
        print("💡 La respuesta del servidor no tiene formato JSON válido")
        
    except Exception as e:
        print(f"\n❌ Error inesperado durante el análisis: {e}")
        print(f"🔍 Tipo de error: {type(e).__name__}")
        
    finally:
        await nc.close()
        print(f"\n🔌 Conexión NATS cerrada")

async def ejemplo_analisis_frame_unico():
    """Ejemplo de análisis de un solo frame."""
    print(f"\n" + "="*70)
    print("🖼️  EJEMPLO: ANÁLISIS DE FRAME ÚNICO")
    print("="*70)
    
    nc = await nats.connect("nats://localhost:4222")
    
    try:
        ruta_imagen = "imagenes/personas.jpg"  # Imagen para frame único
        img = cv2.imread(ruta_imagen)
        
        if img is None:
            print(f"⚠️  No se pudo cargar {ruta_imagen}, usando imagen por defecto...")
            # Crear una imagen de prueba si no existe
            img = create_test_image()
        
        height, width, channels = img.shape
        print(f"📏 Imagen: {width}x{height}x{channels}")
        
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        mensaje = {
            "image": img_base64,
            "metadata": {
                "source": "test_camera",
                "timestamp": datetime.now().isoformat(),
                "location": "test_environment"
            }
        }
        
        print(f"📤 Enviando frame único para análisis...")
        
        response = await nc.request(
            "object_detection.analyze_single_frame",
            json.dumps(mensaje).encode(),
            timeout=30.0
        )
        
        # Mostrar respuesta cruda
        respuesta_cruda = response.data.decode()
        print(f"\n📥 RESPUESTA CRUDA (Frame único):")
        print(f"{'-'*50}")
        print(respuesta_cruda)
        print(f"{'-'*50}")
        
        resultado = json.loads(respuesta_cruda)
        
        if resultado['success']:
            print(f"\n✅ Frame único procesado exitosamente!")
            print(f"📨 {resultado['message']}")
        else:
            print(f"\n❌ Error procesando frame único:")
            print(f"📨 {resultado['message']}")
            
    except Exception as e:
        print(f"❌ Error en análisis de frame único: {e}")
        
    finally:
        await nc.close()

def create_test_image():
    """Crear una imagen de prueba si no existe la imagen real."""
    import numpy as np
    
    # Crear imagen de prueba de 640x480 con texto
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gris
    
    # Agregar texto
    cv2.putText(img, 'TEST IMAGE', (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(img, 'Object Detection Test', (180, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Agregar algunos rectángulos para simular objetos
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 3)
    cv2.rectangle(img, (400, 150), (500, 250), (0, 0, 255), 3)
    
    return img

async def mostrar_informacion_conexion():
    """Mostrar información sobre la conexión NATS."""
    print(f"\n" + "="*70)
    print("🔌 INFORMACIÓN DE CONEXIÓN")
    print("="*70)
    
    try:
        nc = await nats.connect("nats://localhost:4222")
        print("✅ Conexión a NATS exitosa")
        print(f"📡 Servidor: nats://localhost:4222")
        print(f"🆔 Client ID: {nc._client_id}")
        print(f"📊 Estado: Conectado")
        
        # Información adicional si está disponible
        if hasattr(nc, '_server_info'):
            server_info = nc._server_info
            print(f"🖥️  Servidor NATS:")
            print(f"   • Versión: {server_info.get('version', 'N/A')}")
            print(f"   • ID: {server_info.get('server_id', 'N/A')}")
        
        await nc.close()
        print("🔌 Conexión cerrada correctamente")
        
    except Exception as e:
        print(f"❌ Error conectando a NATS: {e}")
        print("💡 Asegúrate de que el servidor NATS esté corriendo")

if __name__ == "__main__":
    print("🚀 EJECUTANDO PRUEBAS CON MICROSERVICIO DE DETECCIÓN")
    print("="*70)
    
    async def ejecutar_todas_las_pruebas():
        # Mostrar información de conexión
        await mostrar_informacion_conexion()
        
        # Prueba con múltiples frames
        await ejemplo_analisis_con_imagen_real()
        
        # Prueba with frame único
        await ejemplo_analisis_frame_unico()
        
        print(f"\n🎉 ¡Todas las pruebas completadas!")
        print("="*70)
    
    asyncio.run(ejecutar_todas_las_pruebas())