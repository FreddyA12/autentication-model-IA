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


@csrf_exempt
@require_http_methods(["POST"])
def authenticate_dual(request):
    """
    API endpoint para autenticación dual (cara + voz)
    
    POST /api/authenticate_dual/
    Body: multipart/form-data con campos 'image' y 'audio'
    """
    try:
        # 1. Validar input
        if 'image' not in request.FILES or 'audio' not in request.FILES:
             return JsonResponse({
                'success': False,
                'error': 'Faltan archivos',
                'message': 'Se requiere imagen y audio'
            }, status=400)
            
        image_file = request.FILES['image']
        audio_file = request.FILES['audio']
        
        print(f"🔄 Iniciando autenticación dual...")
        
        # 2. Procesar Cara
        print("   Procesando cara...")
        face_service = get_face_service()
        image_bytes = image_file.read()
        face_result = face_service.predict(image_bytes)
        
        # 3. Procesar Voz
        print("   Procesando voz...")
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
            
        try:
            voice_result = voice_service.predict(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
                
        # 4. Lógica de Autenticación Dual (con fallback a Face-only)
        face_identity = face_result.get('identity')
        voice_identity = voice_result.get('identity')
        
        # Normalizar identidades para comparación
        face_id_norm = str(face_identity).lower() if face_identity else ""
        voice_id_norm = str(voice_identity).lower() if voice_identity else ""
        
        face_success = face_result.get('success', False) and face_identity != 'DESCONOCIDO' and face_identity is not None
        voice_success = voice_identity != 'unknown' and voice_identity != 'error'
        
        final_success = False
        message = ""
        
        # NUEVA LÓGICA: Priorizar reconocimiento facial
        if face_success and voice_success:
            if face_id_norm == voice_id_norm:
                final_success = True
                message = f"Autenticación dual exitosa: {face_identity}"
            else:
                # Si la cara está muy confiada, permitir acceso aunque la voz no coincida
                if face_result.get('confidence', 0) > 90:
                    final_success = True
                    message = f"Autenticación por rostro: {face_identity} (voz: {voice_identity})"
                else:
                    final_success = False
                    message = f"Identidad no coincide: Cara={face_identity}, Voz={voice_identity}"
        elif face_success and not voice_success:
            # Permitir acceso solo con cara si la confianza es alta
            if face_result.get('confidence', 0) > 90:
                final_success = True
                message = f"Autenticación por rostro: {face_identity}"
            else:
                final_success = False
                message = f"Cara reconocida ({face_identity}), pero voz no reconocida"
        elif not face_success and voice_success:
            message = f"Voz reconocida ({voice_identity}), pero cara no reconocida"
        else:
            message = "No se reconoció ni cara ni voz"
            
        print(f"✅ Resultado Dual: {final_success} ({message})")
            
        return JsonResponse({
            'success': final_success,
            'message': message,
            'face_result': face_result,
            'voice_result': voice_result
        })

    except Exception as e:
        print(f"❌ ERROR DUAL: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': f'Error interno: {str(e)}'
        }, status=500)
