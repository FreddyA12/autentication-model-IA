"""
Views for face authentication app
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import traceback

from .face_service import get_face_service


def index(request):
    """
    Página principal con interfaz de cámara web
    """
    return render(request, 'index.html')


@csrf_exempt
@require_http_methods(["POST"])
def predict_face(request):
    """
    API endpoint para predecir identidad desde una imagen
    
    POST /api/predict/
    Body: multipart/form-data con campo 'image'
    
    Returns:
        JSON con resultado de la predicción
    """
    try:
        # Verificar que se envió una imagen
        if 'image' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No se envió ninguna imagen',
                'message': 'Debes enviar un archivo con el campo "image"'
            }, status=400)
        
        # Obtener la imagen
        image_file = request.FILES['image']
        image_bytes = image_file.read()
        
        print(f"📸 Imagen recibida: {len(image_bytes)} bytes")
        
        # Obtener servicio de reconocimiento facial
        print("🔄 Cargando servicio de reconocimiento...")
        face_service = get_face_service()
        
        # Realizar predicción
        print("🧠 Realizando predicción...")
        result = face_service.predict(image_bytes)
        
        print(f"✅ Resultado: {result}")
        return JsonResponse(result)
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': f'Error al procesar la imagen: {str(e)}'
        }, status=500)
