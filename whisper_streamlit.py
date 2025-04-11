# whisper_streamlit.py

import streamlit as st
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import tempfile
import os
import wave
import datetime
from faster_whisper import WhisperModel

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
recording_flag = {"value": False}  # ← スレッド間で共有可能なフラグ

# === SESSION STATE INIT ===

if "recording_state" not in st.session_state:
    st.session_state.recording_state = False

if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []

# === WHISPER MODEL LOAD ===

@st.cache_resource
def load_model():
    return WhisperModel("base", compute_type="auto")

model = load_model()

# === AUDIO HANDLING ===

def audio_callback(indata, frames, time_info, status):
    if recording_flag["value"]:
        audio_q.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=BLOCKSIZE):
        while recording_flag["value"]:
            time.sleep(0.1)

def save_wav(frames, path):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

def transcribe_audio(frames):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        save_wav(frames, path)
        segments, _ = model.transcribe(path)
        results = []
        for segment in segments:
            ts = str(datetime.timedelta(seconds=int(segment.start)))
            results.append(f"[{ts}] {segment.text}")
        return results
    finally:
        try:
            os.remove(path)
        except Exception as e:
            print(f"⚠️ ファイル削除失敗: {e}")

# === UI CONTROLS ===

col1, col2 = st.columns(2)

if col1.button("⏺️ 録音開始", disabled=st.session_state.recording_state):
    st.session_state.recording_state = True
    recording_flag["value"] = True
    thread = threading.Thread(target=record_audio)
    thread.start()
    st.success("録音中...")

if col2.button("⏹️ 停止", disabled=not st.session_state.recording_state):
    st.session_state.recording_state = False
    recording_flag["value"] = False
    st.warning("録音停止しました。音声処理中...")

    # 音声データ収集
    all_frames = []
    while not audio_q.empty():
        data = audio_q.get()
        all_frames.append(data.tobytes())

    if all_frames:
        transcription = transcribe_audio(all_frames)
        st.session_state.transcriptions.extend(transcription)
        st.success("文字起こし完了！")
    else:
        st.error("⚠️ 音声が録音されていませんでした。")

# === TRANSCRIPTION DISPLAY ===

st.subheader("📝 議事録")
if st.session_state.transcriptions:
    for line in st.session_state.transcriptions:
        st.markdown(f"- {line}")
else:
    st.info("まだ議事録がありません。")
