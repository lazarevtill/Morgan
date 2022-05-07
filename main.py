import os
import wave
import torch
import contextlib
import pyaudio

torch.set_grad_enabled(False)
device = torch.device('cuda')
torch.set_num_threads(8)  # safe optimal value, i.e. 2 CPU cores, does not work properly in colab
symbols = '_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–'
local_file = 'model.jit'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v1_kseniya_16000.jit',
                                   local_file)

if not os.path.isfile('tts_utils.py'):
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/tts_utils.py',
                                   'tts_utils.py')

from tts_utils import apply_tts  # modify these utils and use them your project

model = torch.jit.load('model.jit',
                       map_location=device)
model.eval()
example_batch = ['Я видел: Москва уносила, слышал: «Москва не Россия»']
sample_rate = 16000
model = model.to(device)

audio = apply_tts(texts=example_batch,
                  model=model,
                  sample_rate=sample_rate,
                  symbols=symbols,
                  device=device)


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


for i, _audio in enumerate(audio):
    write_wave(path=f'test_{str(i).zfill(3)}.wav',
               audio=(audio[i] * 32767).numpy().astype('int16'),
               sample_rate=16000)

# define stream chunk
chunk = 1024

# open a wav format music
f = wave.open(r"test_000.wav", "rb")
# instantiate PyAudio
p = pyaudio.PyAudio()
# open stream
stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                channels=f.getnchannels(),
                rate=f.getframerate(),
                output=True)
# read data
data = f.readframes(chunk)

# play stream
while data:
    stream.write(data)
    data = f.readframes(chunk)

# stop stream
stream.stop_stream()
stream.close()

# close PyAudio
p.terminate()
