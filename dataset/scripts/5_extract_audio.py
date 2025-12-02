from pathlib import Path

from moviepy.editor import VideoFileClip

VIDEOS_DIR = Path("dataset/videos")
AUDIO_DIR = Path("dataset/audio_raw")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SAMPLE_RATE = 16_000


def extract_audio(video_path: Path, wav_path: Path) -> None:
    clip = VideoFileClip(str(video_path))
    if clip.audio is None:
        raise RuntimeError(f"El video {video_path.name} no tiene pista de audio.")

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    clip.audio.write_audiofile(
        str(wav_path),
        fps=TARGET_SAMPLE_RATE,
        nbytes=2,
        codec="pcm_s16le",
        verbose=False,
        logger=None,
    )
    clip.close()


def main() -> None:
    if not VIDEOS_DIR.exists():
        raise SystemExit("No se encontró dataset/videos.")

    videos = [
        f for f in sorted(VIDEOS_DIR.iterdir()) if f.suffix.lower() in {".mp4", ".mov", ".avi"}
    ]
    if not videos:
        raise SystemExit("No se encontraron videos de entrada.")

    for video in videos:
        user = video.stem.lower()
        wav_path = AUDIO_DIR / f"{user}.wav"
        print(f"Extrayendo audio de {video.name} -> {wav_path.name}")
        extract_audio(video, wav_path)

    print("Extracción de audio finalizada.")


if __name__ == "__main__":
    main()
