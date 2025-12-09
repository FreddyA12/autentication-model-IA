"""
=============================================================================
PASO 4: PREDICCIÓN DE VOZ - PROBAR MODELO
=============================================================================

Prueba el modelo entrenado con audios del dataset y audios de test.

Pipeline completo:
    audio.wav → ECAPA-TDNN → embedding(192) → MLP → identidad

Prueba:
    1. Audios del dataset (dentro de dataset_voice/)
    2. Audios nuevos (dentro de test_audios/)

Uso:
    python dataset/scripts_voice/4_predict_voice.py
    
El script probará automáticamente:
    - Algunos audios del dataset (para verificar que reconoce bien)
    - Todos los audios en test_audios/ (audios nuevos de prueba)
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import tensorflow as tf

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
MODELS_DIR = Path("dataset/models")
MODEL_PATH = MODELS_DIR / "voice_mlp_best.keras"
CLASS_INDICES_PATH = MODELS_DIR / "voice_class_indices.json"

DATASET_VOICE_DIR = Path("dataset/dataset_voice")
TEST_AUDIOS_DIR = Path("dataset/test_audios")

SAMPLE_RATE = 16000
CONFIDENCE_THRESHOLD = 0.80  # Ajustado a 80% para mejor detección de desconocidos


class VoicePredictor:
    """
    Predictor de identidad por voz
    
    Usa ECAPA-TDNN + MLP entrenado
    """
    
    def __init__(self, model_path, class_indices_path):
        """
        Inicializa el predictor
        
        Args:
            model_path: Ruta al modelo MLP entrenado
            class_indices_path: Ruta al mapeo de clases
        """
        print("📦 Inicializando predictor...")
        
        # Cargar ECAPA-TDNN
        print("   Cargando ECAPA-TDNN...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        
        # Cargar MLP
        print("   Cargando MLP...")
        self.mlp = tf.keras.models.load_model(model_path)
        
        # Cargar mapeo de clases
        with open(class_indices_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
            self.label_map = {int(k): v for k, v in self.label_map.items()}
        
        print(f"   ✅ Predictor listo ({len(self.label_map)} clases)")
        print(f"   Dispositivo: {device}\n")
    
    def extract_embedding(self, audio_path):
        """
        Extrae embedding de audio usando ECAPA-TDNN
        
        Args:
            audio_path: Ruta al archivo de audio
        
        Returns:
            embedding: Vector de 192 dimensiones
        """
        # Cargar audio
        signal, fs = torchaudio.load(str(audio_path))
        
        # Convertir a mono si es estéreo
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        # Resamplear a 16kHz si es necesario
        if fs != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(fs, SAMPLE_RATE)
            signal = resampler(signal)
        
        # Extraer embedding
        with torch.no_grad():
            embedding = self.ecapa.encode_batch(signal)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    
    def predict(self, audio_path):
        """
        Predice la identidad desde un audio
        
        Args:
            audio_path: Ruta al archivo de audio
        
        Returns:
            dict: Resultado con identity, confidence, probabilities
        """
        # Extraer embedding
        embedding = self.extract_embedding(audio_path)
        embedding = np.expand_dims(embedding, axis=0)  # (1, 192)
        
        # Clasificar
        predictions = self.mlp.predict(embedding, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        
        # Determinar identidad
        if confidence >= CONFIDENCE_THRESHOLD:
            identity = self.label_map[predicted_idx]
        else:
            identity = "unknown"
        
        # Construir resultado
        result = {
            'identity': identity,
            'confidence': confidence,
            'probabilities': {
                self.label_map[i]: float(predictions[i])
                for i in range(len(predictions))
            }
        }
        
        return result


def print_result(result, audio_path):
    """
    Imprime el resultado de forma visual
    
    Args:
        result: Diccionario con el resultado
        audio_path: Ruta al audio analizado
    """
    print("\n" + "="*70)
    print("📊 RESULTADOS DE PREDICCIÓN")
    print("="*70)
    print(f"\n🎤 Audio: {audio_path}")
    
    print("\n📈 Probabilidades por clase:")
    
    # Ordenar por probabilidad descendente
    probs_sorted = sorted(
        result['probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for person, prob in probs_sorted:
        bar_length = int(prob * 50)
        bar = "█" * bar_length
        print(f"   {person:12s} {prob*100:6.2f}%  {bar}")
    
    print(f"\n{'='*70}")
    
    if result['identity'] != 'unknown':
        print(f"✅ IDENTIDAD: {result['identity'].upper()}")
        print(f"   Confianza: {result['confidence']*100:.2f}%")
    else:
        print(f"❌ DESCONOCIDO")
        print(f"   Confianza máxima: {result['confidence']*100:.2f}%")
        print(f"   Umbral requerido: {CONFIDENCE_THRESHOLD*100:.0f}%")
    
    print("="*70 + "\n")


def test_dataset_samples(predictor):
    """
    Prueba con algunos audios del dataset (audios conocidos)
    """
    print("\n" + "="*70)
    print("🎯 PRUEBA 1: AUDIOS DEL DATASET (audios conocidos)")
    print("="*70)
    print("Probando que el modelo reconoce correctamente los audios de entrenamiento\n")
    
    if not DATASET_VOICE_DIR.exists():
        print(f"⚠️  No se encontró {DATASET_VOICE_DIR}")
        return
    
    # Obtener todas las personas
    persons = sorted([d for d in DATASET_VOICE_DIR.iterdir() if d.is_dir()])
    
    if not persons:
        print("⚠️  No hay carpetas en dataset_voice/")
        return
    
    results = []
    
    for person_dir in persons:
        person_name = person_dir.name
        audios = sorted(person_dir.glob("*.wav"))
        
        if not audios:
            continue
        
        # Tomar el primer audio de cada persona
        audio_path = audios[0]
        
        print(f"🎤 Probando: {person_name}/{audio_path.name}")
        
        try:
            result = predictor.predict(audio_path)
            predicted = result['identity']
            confidence = result['confidence'] * 100
            
            # Verificar si la predicción es correcta
            is_correct = (predicted.lower() == person_name.lower())
            
            if is_correct:
                print(f"   ✅ Correcto: {predicted} ({confidence:.1f}%)\n")
            else:
                print(f"   ❌ Error: predijo {predicted} ({confidence:.1f}%), esperaba {person_name}\n")
            
            results.append({
                'audio': f"{person_name}/{audio_path.name}",
                'expected': person_name,
                'predicted': predicted,
                'confidence': confidence,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"   ❌ Error al procesar: {e}\n")
    
    # Resumen
    if results:
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = (correct / total) * 100
        
        print("\n" + "="*70)
        print("📊 RESUMEN - AUDIOS DEL DATASET")
        print("="*70)
        print(f"Correctas: {correct}/{total}")
        print(f"Accuracy: {accuracy:.1f}%")
        print("="*70 + "\n")


def test_new_audios(predictor):
    """
    Prueba con audios nuevos (fuera del dataset)
    """
    print("\n" + "="*70)
    print("🆕 PRUEBA 2: AUDIOS DE TEST (audios nuevos)")
    print("="*70)
    print(f"Probando audios en {TEST_AUDIOS_DIR}/\n")
    
    # Crear directorio si no existe
    if not TEST_AUDIOS_DIR.exists():
        TEST_AUDIOS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ Creada carpeta: {TEST_AUDIOS_DIR}")
        print(f"   Coloca audios nuevos aquí para probar el modelo\n")
        print("💡 Estructura recomendada:")
        print("   test_audios/")
        print("      alison_test1.wav")
        print("      freddy_test1.opus")
        print("      desconocido1.mp3")
        print("\n📝 Formatos soportados: .wav, .opus, .mp3, .ogg, .m4a")
        print("⏱️  Duración recomendada: 5-10 segundos por audio")
        print("="*70 + "\n")
        return
    
    # Buscar audios de test (múltiples formatos)
    test_audios = []
    for ext in ['*.wav', '*.opus', '*.mp3', '*.ogg', '*.m4a']:
        test_audios.extend(TEST_AUDIOS_DIR.glob(ext))
    test_audios = sorted(test_audios)
    
    if not test_audios:
        print("⚠️  No hay archivos de audio en test_audios/")
        print(f"   Coloca audios en {TEST_AUDIOS_DIR}/ para probar")
        print("   Formatos soportados: .wav, .opus, .mp3, .ogg, .m4a\n")
        return
    
    print(f"Encontrados {len(test_audios)} audios de test\n")
    
    for audio_path in test_audios:
        print(f"🎤 Probando: {audio_path.name}")
        
        try:
            result = predictor.predict(audio_path)
            predicted = result['identity']
            confidence = result['confidence'] * 100
            
            # Mostrar resultado
            if predicted != 'unknown':
                print(f"   ✅ Identificado como: {predicted.upper()} ({confidence:.1f}%)")
            else:
                print(f"   ⚠️  Voz DESCONOCIDA (confianza máxima: {confidence:.1f}%)")
            
            # Mostrar top 3 probabilidades
            probs_sorted = sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            print("   Top 3 probabilidades:")
            for person, prob in probs_sorted:
                print(f"      {person:12s} {prob*100:6.2f}%")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error al procesar: {e}\n")
    
    print("="*70 + "\n")


def main():
    print("="*70)
    print("PASO 4: PROBAR MODELO DE VOZ")
    print("="*70 + "\n")
    
    print("""
    Este script prueba tu modelo entrenado con:
    
    1️⃣  Audios del dataset (para verificar accuracy)
    2️⃣  Audios nuevos en test_audios/ (audios de prueba)
    
    """)
    
    # Verificar que existe el modelo
    if not MODEL_PATH.exists():
        print(f"❌ No se encontró el modelo: {MODEL_PATH}")
        print("   Ejecuta primero: python dataset/scripts_voice/3_train_voice_mlp.py")
        return
    
    # Crear predictor
    predictor = VoicePredictor(MODEL_PATH, CLASS_INDICES_PATH)
    
    # Prueba 1: Audios del dataset
    test_dataset_samples(predictor)
    
    # Prueba 2: Audios de test
    test_new_audios(predictor)
    
    print("\n✅ Pruebas completadas\n")
    print("💡 Tips:")
    print("   - Coloca más audios en test_audios/ para seguir probando")
    print("   - Formatos: .wav, .opus, .mp3, .ogg, .m4a")
    print("   - Duración: al menos 5 segundos (idealmente 5-10s)")
    print("   - Calidad: habla clara, sin mucho ruido de fondo")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
