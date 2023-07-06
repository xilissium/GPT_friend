import openai
import pyaudio
import math
import struct
import wave
import time
import os
import whisper
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play
import GUI



################SETINGS##############
audio_sensi = 18        ## choos your audio sensibilty
audio_end_rec = 1       ##in second
audio_file = r'YOUR FILE PATH'
personalité = ""        ## set the personnality of your assitant 
content = ""            ## add more detail for your assitant
whisper_model = "base" ## tiny , base , small , medium , large (it's the model for th speech_to text (need to be download once a first start) 
AI_language = "fr"     ## select ryour language for Speech_to_text it help tu run faster (ex en,de)
key_api = 'YOUR API KEY' ## your api key from openai
voice_model = 'tts_models/fr/css10/vits' ##select voire model from TTS api this model is french only 
IA_model = "gpt-3.5-turbo"     ##chose the ia model on openai site
use_gpu = False                ## use or no of the gpu to run faster (maybe don't work) experimentale
#####################################


openai.api_key = key_api
model = whisper.load_model(whisper_model)
tts = TTS(model_name=voice_model, progress_bar=True, gpu=use_gpu)

Threshold = audio_sensi
SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
swidth = 2
TIMEOUT_LENGTH = audio_end_rec
audio_name = 0
state = 0
context = ""
audio_output = "file.wav"

def text_to_speech(text):
    global audio_output
    tts.tts_to_file(text, file_path=audio_output)
    ##song = AudioSegment.from_file(audio_output)
    ##play(song)
    GUI.start()
    

def recognition(f):
    
    result = model.transcribe(f, language=AI_language, fp16=False)
    os.remove(f)
    return result['text']

def generer_reponse(question,context):
    global personalité
    global content
    response = openai.ChatCompletion.create(
        model=IA_model,  
        messages=[
            {"role": "system", "content": personalité},
            {"role": "user", "content": question},
            {"role": "assistant", "content": content},
            {"role": "assistant", "content": context}
        ]
    )

    if response and response.choices:
        return response.choices[0].message.content.strip()
    else:
        return "Je suis désolé, je n'ai pas pu générer de réponse."

class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=chunk
                                  )

    def record(self):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(chunk)
            if self.rms(data) >= Threshold: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        self.write(b''.join(rec))

    def write(self, recording):
        global audio_name
        n_files = len(os.listdir(audio_file))
        audio_name = n_files
        filename = os.path.join(audio_file, '{}.wav'.format(n_files))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        
        print('Written to file: {}'.format(filename))
        
        
    
    
    def listen(self):
        global state
        print('Listening beginning')
        while state != 1:
            input = self.stream.read(chunk)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                self.record()
                state = 1

speech_to_text = Recorder()

speech_to_text.listen()

contexte = ""
while True:
    file = str(audio_name)+'.wav'
    question = recognition(file)
    print(question)
    if question == "Merci.":
        text_to_speech("pas de soucis")
        break
    reponse = generer_reponse(question, contexte)
    print("Réponse : ", reponse)
    
    text_to_speech(reponse)
    state = 0
    speech_to_text.listen()
    contexte += "Q: " + question + "\nA: " + reponse + "\n"
