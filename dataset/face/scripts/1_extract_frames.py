import cv2
import os

VIDEOS_DIR = "dataset/face/videos"
RAW_DIR = "dataset/face/processed"
TARGET_WIDTH = 800
FRAME_STEP = 5

os.makedirs(RAW_DIR, exist_ok=True)

def procesar_video(path, user):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error {path}")
        return

    save_dir = os.path.join(RAW_DIR, user)
    os.makedirs(save_dir, exist_ok=True)

    frame_id = 0
    saved = 0

    print(f"Procesando {user}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_STEP == 0:
            h, w = frame.shape[:2]
            scale = TARGET_WIDTH / w
            nh = int(h * scale)
            frame = cv2.resize(frame, (TARGET_WIDTH, nh))
            name = f"{user}_{saved:05d}.jpg"
            cv2.imwrite(os.path.join(save_dir, name), frame)
            saved += 1

        frame_id += 1

    cap.release()
    print(f"{user}: {saved} frames guardados")


if __name__ == "__main__":
    if not os.path.exists(VIDEOS_DIR):
        print("No existe dataset/videos")
        exit()

    videos = [v for v in os.listdir(VIDEOS_DIR) if v.lower().endswith(('.mp4','.mov','.avi'))]

    print(f"Videos encontrados: {len(videos)}")

    for v in videos:
        user = os.path.splitext(v)[0].lower()
        procesar_video(os.path.join(VIDEOS_DIR, v), user)

    print("Extracción finalizada")
