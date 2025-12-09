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
from .voice_service import voice_service


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


@csrf_exempt
@require_http_methods(["POST"])
def predict_voice(request):
    """
    API endpoint para predecir identidad desde un audio
    
    POST /api/predict_voice/
    Body: multipart/form-data con campo 'audio'
    
    Returns:
        JSON con resultado de la predicción
    """
    try:
        # Verificar que se envió un audio
        if 'audio' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No se envió ningún audio',
                'message': 'Debes enviar un archivo con el campo "audio"'
            }, status=400)
        

        
        # Obtener el audio (ahora es WAV directo del navegador)
        audio_file = request.FILES['audio']
        
        print(f"🎤 Audio recibido: {audio_file.name}, {audio_file.size} bytes")
        
        # Guardar temporalmente
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        print(f"✅ Audio guardado en: {tmp_path}")
        
        try:
            # Realizar predicción
            print("🧠 Realizando predicción de voz...")
            result = voice_service.predict(tmp_path)
            
            # Agregar success flag
            result['success'] = result.get('identity') != 'error'
            
            print(f"✅ Resultado: {result}")
            return JsonResponse(result)
        
        finally:
            # Eliminar archivo temporal
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': f'Error al procesar el audio: {str(e)}'
        }, status=500)
