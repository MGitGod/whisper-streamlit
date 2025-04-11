# whisper_streamlit.py

import streamlit as st
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import tempfile
import os
from faster_whisper import WhisperModel
import datetime
import wave

# === SETUP ===

st.set_page_config(page_title="Whisper 議事録", layout="centered")

st.title("🎙️ Whisper リアルタイム議事録")
st.caption("録音 → 自動文字起こし → タイムスタンプ付きで表示")

# === PARAMETERS ===

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1024

# === GLOBALS ===

audio_q = queue.Queue()
recording = False
transcriptions = []
audio_buffer = []

# === INIT WHISPER ===

@st.cache_resource
def load_model():
    return WhisperModel("base", compute_type="auto")

model = load_model()

# === AUDIO HANDLING ===

def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_q.put(indata.copy())

def record_audio():
    global recording
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=BLOCKSIZE):
        while recording:
            time.sleep(0.1)

def save_wav(frames, path):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

def transcribe_audio(frames):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        save_wav(frames, f.name)
        segments, _ = model.transcribe(f.name)
        results = []
        for segment in segments:
            ts = str(datetime.timedelta(seconds=int(segment.start)))
            results.append(f"[{ts}] {segment.text}")
        os.remove(f.name)
        return results

# === STREAMLIT UI ===

if "recording_state" not in st.session_state:
    st.session_state.recording_state = False

col1, col2 = st.columns(2)

if col1.button("⏺️ 録音開始", disabled=st.session_state.recording_state):
    st.session_state.recording_state = True
    recording = True
    audio_buffer.clear()
    thread = threading.Thread(target=record_audio)
    thread.start()
    st.success("録音中...")

if col2.button("⏹️ 停止", disabled=not st.session_state.recording_state):
    st.session_state.recording_state = False
    recording = False
    st.warning("録音停止しました。音声処理中...")

    # Collect frames
    all_frames = []
    while not audio_q.empty():
        data = audio_q.get()
        all_frames.append(data.tobytes())

    transcription = transcribe_audio(all_frames)
    transcriptions.extend(transcription)
    st.success("文字起こし完了！")

# === TRANSCRIPT DISPLAY ===

st.subheader("📝 議事録")
if transcriptions:
    for line in transcriptions:
        st.markdown(f"- {line}")
else:
    st.info("まだ議事録がありません。")
