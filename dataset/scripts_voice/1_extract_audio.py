"""
=============================================================================
PASO 1: EXTRAER AUDIO DE VIDEOS
=============================================================================

Extrae audio de videos y los guarda en dataset_voice/persona/audioX.wav

Este es el equivalente a 1_extract_frames.py pero para audio.

Uso:
    python dataset/scripts_voice/1_extract_audio.py

Estructura de salida:
    dataset/dataset_voice/
        freddy/
            freddy_001.wav
            freddy_002.wav
        melanie/
            melanie_001.wav
        rafael/
        william/
        ismael/
"""

import os
from pathlib import Path
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
VIDEOS_DIR = Path("dataset/videos")
OUTPUT_DIR = Path("dataset/dataset_voice")
TARGET_SAMPLE_RATE = 16000  # 16 kHz requerido por ECAPA-TDNN
MIN_DURATION = 3.0          # Mínimo 3 segundos (aumentado)
MAX_DURATION = 10.0         # Máximo 10 segundos (aumentado)


def extract_audio_segments(video_path, person_name):
    """
    Extrae audio del video y lo guarda en segmentos
    
    Args:
        video_path: Ruta al video
        person_name: Nombre de la persona (ej: 'freddy')
    """
    clip = None
    try:
        clip = VideoFileClip(str(video_path))
        
        if clip.audio is None:
            print(f"   ⚠️  {video_path.name} no tiene audio")
            if clip:
                clip.close()
            return 0
        
        # Crear directorio de la persona
        person_dir = OUTPUT_DIR / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Obtener duración del audio
        duration = clip.duration
        print(f"   Duración: {duration:.1f}s")
        
        if duration < MIN_DURATION:
            print(f"   ⚠️  Audio muy corto")
            clip.close()
            return 0
        
        # Configuración optimizada para máxima extracción
        segment_duration = 5.0  # Aumentado a 5s (mejor para ECAPA-TDNN)
        hop_duration = 2.0      # Solapamiento de 60% (5s con salto de 2s)
        
        # Calcular número de segmentos con solapamiento
        num_segments = int((duration - segment_duration) / hop_duration) + 1
        num_segments = max(1, num_segments)  # Al menos 1 segmento
        print(f"   Extrayendo {num_segments} segmentos de {segment_duration}s...")
        
        # Extraer múltiples segmentos con solapamiento
        segments_saved = 0
        for i in range(num_segments):
            start_time = i * hop_duration
            end_time = start_time + segment_duration
            
            # No pasar del final del video
            if end_time > duration:
                end_time = duration
                start_time = max(0, end_time - segment_duration)
            
            # Verificar duración mínima
            if end_time - start_time < MIN_DURATION:
                continue
            
            audio_path = person_dir / f"{person_name}_{segments_saved+1:03d}.wav"
            
            try:
                # Extraer segmento
                segment = clip.subclip(start_time, end_time)
                segment.audio.write_audiofile(
                    str(audio_path),
                    fps=TARGET_SAMPLE_RATE,
                    nbytes=2,
                    codec='pcm_s16le',
                    verbose=False,
                    logger=None
                )
                segment.close()
                del segment
                segments_saved += 1
            except Exception as seg_error:
                # Ignorar errores individuales de segmentos
                pass
        
        clip.close()
        del clip
        return segments_saved
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}")
        if clip:
            try:
                clip.close()
            except:
                pass
        return 0


def main():
    print("="*70)
    print("PASO 1: EXTRAER AUDIO DE VIDEOS")
    print("="*70)
    print("""
    Este script extrae audio de tus videos y crea el dataset de voz:
    
    dataset/dataset_voice/
        freddy/
            freddy_001.wav  (3s, 16kHz, mono)
            freddy_002.wav
        melanie/
            melanie_001.wav
        ...
    
    Requisitos:
        ✔️ Videos en dataset/videos/
        ✔️ Nombre del video = nombre de la persona
        ✔️ Audio de al menos 2-5 segundos
    """)
    
    # Verificar directorio de videos
    if not VIDEOS_DIR.exists():
        print(f"❌ No existe el directorio: {VIDEOS_DIR}")
        return
    
    # Buscar videos
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI']
    videos = []
    for ext in video_extensions:
        videos.extend(VIDEOS_DIR.glob(f'*{ext}'))
    
    if not videos:
        print(f"❌ No se encontraron videos en {VIDEOS_DIR}")
        return
    
    print(f"\n📹 Encontrados {len(videos)} videos\n")
    
    # Crear directorio de salida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_segments = 0
    
    for video_path in tqdm(videos, desc="Procesando videos"):
        # El nombre del video es el nombre de la persona
        person_name = video_path.stem.lower()
        
        print(f"\n👤 Procesando: {person_name}")
        print(f"   Video: {video_path.name}")
        
        segments = extract_audio_segments(video_path, person_name)
        
        if segments > 0:
            print(f"   ✅ Extraídos {segments} segmentos de audio")
            total_segments += segments
        else:
            print(f"   ❌ No se pudo extraer audio")
    
    print("\n" + "="*70)
    print("📊 RESUMEN")
    print("="*70)
    
    # Contar audios por persona
    if OUTPUT_DIR.exists():
        persons = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
        print(f"\n✅ Total de personas: {len(persons)}")
        print(f"✅ Total de segmentos: {total_segments}\n")
        
        print("Audios por persona:")
        for person_dir in sorted(persons):
            audio_files = list(person_dir.glob("*.wav"))
            print(f"   {person_dir.name:12s}: {len(audio_files)} audios")
    
    print("\n" + "="*70)
    print("✅ PASO 1 COMPLETADO")
    print("="*70)
    print(f"""
    Tus audios están en: {OUTPUT_DIR}/
    
    Siguiente paso:
        python dataset/scripts_voice/2_generate_voice_embeddings.py
    """)


if __name__ == "__main__":
    main()
