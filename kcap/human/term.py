"""Terminal input helpers for the owner-facing CLIs."""


def read_multiline(prompt: str) -> str:
    """Read lines until an empty line. Returns '/done' if the user types it."""
    print(prompt + "  (finish with an empty line)")
    lines = []
    while True:
        line = input()
        if line.strip() == "/done":
            return "/done"
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def ask_choice(prompt: str, choices: list[str]) -> str:
    while True:
        a = input(f"{prompt} [{'/'.join(choices)}] ").strip().lower()
        if a in choices:
            return a


def record_and_transcribe(model_size: str = "small") -> str:
    """Optional voice answer: press Enter to start and again to stop. Needs faster-whisper + sounddevice."""
    import threading

    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel

    frames, stop = [], threading.Event()
    input("Press Enter to start recording...")

    def cb(indata, *_):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=cb):
        threading.Thread(target=lambda: (input("Recording... press Enter to stop."), stop.set()), daemon=True).start()
        stop.wait()
    audio = np.concatenate(frames)[:, 0] if frames else np.zeros(1, dtype="float32")
    segs, _ = WhisperModel(model_size, compute_type="int8").transcribe(audio)
    text = " ".join(s.text.strip() for s in segs)
    print(f"[transcribed] {text}")
    return text
