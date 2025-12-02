import os
import random

CLEAN_DIR = "dataset/dataset_clean"

def balance_dataset():
    """Balancea el dataset para que todas las personas tengan el mismo número de imágenes"""
    
    # Primero, contar imágenes por persona
    person_counts = {}
    for person in os.listdir(CLEAN_DIR):
        person_dir = os.path.join(CLEAN_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        person_counts[person] = len(images)
    
    if not person_counts:
        print("No se encontraron personas en dataset_clean")
        return
    
    # Encontrar el mínimo
    min_count = min(person_counts.values())
    max_count = max(person_counts.values())
    
    print("\n📊 Estado actual del dataset:")
    print("="*50)
    for person, count in sorted(person_counts.items()):
        print(f"  {person:<15}: {count:>4} imágenes")
    print("="*50)
    print(f"\nMínimo: {min_count} | Máximo: {max_count}")
    
    if min_count == max_count:
        print("\n✅ El dataset ya está balanceado!")
        return
    
    print(f"\n🎯 Balanceando a {min_count} imágenes por persona...")
    
    # Balancear
    for person in person_counts:
        person_dir = os.path.join(CLEAN_DIR, person)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if len(images) > min_count:
            # Eliminar imágenes aleatorias (excepto las últimas para mantener variedad)
            random.shuffle(images)
            to_remove = images[min_count:]
            
            for img in to_remove:
                os.remove(os.path.join(person_dir, img))
            
            print(f"  {person}: Eliminadas {len(to_remove)} imágenes ({len(images)} → {min_count})")
        else:
            print(f"  {person}: Sin cambios ({len(images)} imágenes)")
    
    print("\n✅ Dataset balanceado correctamente!")
    print("🔄 Ahora ejecuta: python dataset/scripts/3_train_model.py")

if __name__ == "__main__":
    balance_dataset()
