import whisper
from datetime import timedelta

def transcribe_with_timestamps(audio_path: str, model_name: str = "base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, word_timestamps=True)  

    segments = []
    for seg in result["segments"]:
        start = str(timedelta(seconds=int(seg["start"])))
        end = str(timedelta(seconds=int(seg["end"])))
        text = seg["text"].strip()
        timestamp = f"{start}-{end}"
        segments.append({
            "text": text,
            "start": seg["start"],
            "end": seg["end"],
            "timestamp": timestamp
        })
    return segments, result["text"]