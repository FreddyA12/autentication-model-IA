"""
=============================================================================
PASO 4: PREDICCIÓN DE VOZ
=============================================================================

Predice la identidad de una persona desde un archivo de audio.

Pipeline completo:
    audio.wav → ECAPA-TDNN → embedding(192) → MLP → identidad

Uso:
    python dataset/scripts_voice/4_predict_voice.py <audio.wav>
    
Ejemplo:
    python dataset/scripts_voice/4_predict_voice.py test_audio.wav
    python dataset/scripts_voice/4_predict_voice.py dataset/dataset_voice/freddy/freddy_001.wav
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

SAMPLE_RATE = 16000
CONFIDENCE_THRESHOLD = 0.50


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


def main():
    print("="*70)
    print("PREDICCIÓN DE VOZ")
    print("="*70 + "\n")
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("❌ Error: Debes proporcionar un archivo de audio\n")
        print("Uso:")
        print("   python dataset/scripts_voice/4_predict_voice.py <audio.wav>\n")
        print("Ejemplos:")
        print("   python dataset/scripts_voice/4_predict_voice.py test.wav")
        print("   python dataset/scripts_voice/4_predict_voice.py dataset/dataset_voice/freddy/freddy_001.wav")
        return
    
    audio_path = Path(sys.argv[1])
    
    # Verificar que existe el audio
    if not audio_path.exists():
        print(f"❌ No se encontró el archivo: {audio_path}")
        return
    
    # Verificar que existe el modelo
    if not MODEL_PATH.exists():
        print(f"❌ No se encontró el modelo: {MODEL_PATH}")
        print("   Ejecuta primero: python dataset/scripts_voice/3_train_voice_mlp.py")
        return
    
    # Crear predictor
    predictor = VoicePredictor(MODEL_PATH, CLASS_INDICES_PATH)
    
    # Predecir
    print(f"🔍 Analizando audio: {audio_path.name}...")
    result = predictor.predict(audio_path)
    
    # Mostrar resultado
    print_result(result, audio_path.name)
    
    # Mostrar JSON
    print("📄 Resultado en formato JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
