# src/response_handler.py
from datetime import datetime
from typing import Dict, Any

class ResponseHandler:
    """Clase para manejar las respuestas del microservicio."""
    
    def create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Crear respuesta exitosa con datos completos."""
        return {
            'success': True,
            'timestamp': self._get_timestamp(),
            'data': data,
            'error': None
        }
    
    def create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Crear respuesta de error con datos completos."""
        return {
            'success': False,
            'timestamp': self._get_timestamp(),
            'data': None,
            'error': {
                'message': error_message,
                'type': 'processing_error'
            }
        }
    
    def create_simple_success_response(self, message: str) -> Dict[str, Any]:
        """Crear respuesta simple de éxito para NATS."""
        return {
            'success': True,
            'timestamp': self._get_timestamp(),
            'message': message,
            'status': 'processed'
        }
    
    def create_simple_error_response(self, error_message: str) -> Dict[str, Any]:
        """Crear respuesta simple de error para NATS."""
        return {
            'success': False,
            'timestamp': self._get_timestamp(),
            'message': error_message,
            'status': 'error'
        }
    
    def _get_timestamp(self) -> str:
        """Obtener timestamp actual en formato ISO."""
        return datetime.utcnow().isoformat()