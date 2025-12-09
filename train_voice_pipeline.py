"""
=============================================================================
PIPELINE COMPLETO: ENTRENAR MODELO DE VOZ
=============================================================================

Este script ejecuta todos los pasos necesarios para entrenar el modelo de voz:

1. Extrae audio de los videos (si no existen)
2. Extrae embeddings con wav2vec 2.0
3. Entrena el clasificador
4. Evalúa el modelo

Uso:
    python train_voice_pipeline.py
"""

import subprocess
import sys
import os
from pathlib import Path

SCRIPTS_DIR = Path("dataset/scripts")

def run_script(script_name):
    """Ejecuta un script de Python"""
    script_path = SCRIPTS_DIR / script_name
    
    print("\n" + "="*70)
    print(f"🚀 EJECUTANDO: {script_name}")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error ejecutando {script_name}")
        return False
    
    return True


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║          🎤 PIPELINE DE ENTRENAMIENTO DE VOZ 🎤                   ║
║                                                                   ║
║  Este script ejecuta automáticamente:                            ║
║    1. Extracción de audio                                        ║
║    2. Extracción de embeddings (wav2vec 2.0)                     ║
║    3. Entrenamiento del clasificador                             ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Verificar que existen videos
    videos_dir = Path("dataset/videos")
    if not videos_dir.exists() or not list(videos_dir.glob("*.mp4")):
        print("❌ No se encontraron videos en dataset/videos/")
        print("   Coloca los videos de cada persona en esa carpeta")
        return
    
    # Paso 1: Extraer audio
    audio_dir = Path("dataset/audio_raw")
    if not audio_dir.exists() or not list(audio_dir.glob("*.wav")):
        print("\n📌 Paso 1: Extrayendo audio de videos...")
        if not run_script("5_extract_audio.py"):
            return
    else:
        print("\n✅ Audio ya extraído, saltando paso 1")
    
    # Paso 2: Extraer embeddings
    embeddings_path = Path("dataset/embeddings/voice_embeddings.pkl")
    print("\n📌 Paso 2: Extrayendo embeddings con wav2vec 2.0...")
    if not run_script("6_extract_voice_embeddings.py"):
        return
    
    # Paso 3: Entrenar clasificador
    print("\n📌 Paso 3: Entrenando clasificador...")
    if not run_script("7_train_voice_classifier.py"):
        return
    
    # Final
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETADO")
    print("="*70)
    print("""
    Tu modelo de voz está listo en:
        dataset/models/voice_classifier_best.keras
        
    Para probar el modelo:
        python dataset/scripts/8_predict_voice.py <audio.wav>
    """)


if __name__ == "__main__":
    main()
