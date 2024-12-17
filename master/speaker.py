import os
import wave
import pyaudio
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
from pathlib import Path
import openai

script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
base_path=parent_directory+"/data/"
# 设置你的 OpenAI API 密钥
openai.api_key = ''
speech_file_path = base_path +"speech.mp3"
def get_file(str1):
    response = openai.audio.speech.create(
      model="tts-1",
      voice="alloy",
      input=str1
    
    )
    response.stream_to_file(speech_file_path)


def play(speech_file_path):
    try:
        sound = AudioSegment.from_file(speech_file_path)
    except CouldntDecodeError:
        print(f"Error: Could not decode {speech_file_path}. Unsupported format or corrupt file.")
        return

    # Convert any format to WAV (if not already WAV)
    if sound.frame_rate != 44100 or sound.sample_width != 2 or sound.channels != 2:
        sound = sound.set_frame_rate(44100).set_sample_width(2).set_channels(2)

    wav_path = base_path+"temp.wav"  # Temporary WAV file path
    sound.export(wav_path, format="wav")

    # Playback WAV file
    chunk = 1024  
    wf = wave.open(wav_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), 
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(), 
                    output=True)
    data = wf.readframes(chunk)  # Read data
    while data != b'':  # Play
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()  # Stop streaming
    stream.close()
    p.terminate()  # Close PyAudio

    # Clean up: delete temporary WAV file
    os.remove(wav_path)

def speak_str(str1):
    get_file(str1)
    play(speech_file_path)

#speak("hello my friend!")




