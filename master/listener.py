import time
from pynput import keyboard
import pyaudio
import wave
import threading
import whisper
from master_order import *

model = whisper.load_model("large-v3")
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
base_path=parent_directory+"/data/"
# 录音参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio = pyaudio.PyAudio()

# 创建一个事件对象
stop_event = threading.Event()

# 全局标志，用于记录状态
RECORDING1 = False

def record_audio(file_name):
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []

    print("开始录音...")
    while not stop_event.is_set():
        data = stream.read(CHUNK)
        frames.append(data)
    print("录音结束...")

    stream.stop_stream()
    stream.close()

    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # 录音完成后调用转录函数
    user_content=transcribe_audio(file_name)
    exec_process(user_content)
    print("按 's' 键开始/停止录音...")

def on_press(key):
    global RECORDING1  # 声明RECORDING1为全局变量
    try:
        if key.char == 's':
            if not RECORDING1:
                RECORDING1 = True
                stop_event.clear()
                threading.Thread(target=record_audio, args=(base_path+"output.wav",)).start()
            else:
                RECORDING1 = False
                stop_event.set()
    except AttributeError:
        pass

def listen_for_key():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def transcribe_audio(file_name):
    
    result = model.transcribe(file_name,language='zh')
    #ssssresult = model.transcribe(file_name,language='en')
    print(result["text"]
)
    return result["text"]

if __name__ == "__main__":
    print("按 's' 键开始/停止录音...")
    listen_for_key()
    audio.terminate()
