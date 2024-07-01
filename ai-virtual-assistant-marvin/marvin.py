import io
import os
import wave
import base64
import librosa
import pyaudio
import numpy as np
import tensorflow.keras.backend as K  # Modified import statement


from speech import langmodel
from marvinChatter import marvinChat

terminate = False

class Application:
    def __init__(self):
        self.startProcess = True
        self.rate = 16000
        self.chunk = int(self.rate / 20)

    def readFromStream(self, stream):
        bytesData = stream.read(self.chunk)
        data = np.fromstring(bytesData, dtype=np.int16)
        return data, bytesData

    def writeWav(self, filename, data, sample_size):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_size)
            wf.setframerate(self.rate)
            wf.writeframes(data)

    def getBase64(self, file_path):
        with io.open(file_path, "rb") as f:
            content = f.read()
        base64_data = base64.b64encode(content)
        return base64_data.decode("utf-8")

    def run(self):
        global terminate
        speechModel = langmodel.speechModel()
        mChat = marvinChat()

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        sample_size = p.get_sample_size(pyaudio.paInt16)

        stopNow = False
        recordNow = False
        highPitch = False
        chunkAdded = False
        lang = "en"

        datasec = []
        tobeCheck = []
        tobeSent = []
        data_chunk = []
        bytes_chunk = []

        mChat.justPlay("audios/ready.wav")

        while not stopNow:
            data, bytesData = self.readFromStream(stream)

            if max(data) > 2500:
                highPitch = True

            if highPitch and self.startProcess:
                if not chunkAdded:
                    if len(bytes_chunk) > 0:
                        datasec.extend(data_chunk)
                        tobeCheck.append(bytes_chunk)
                        tobeSent.append(bytes_chunk)
                    chunkAdded = True

                tobeCheck.append(bytesData)
                if not recordNow:
                    datasec.extend(data)
                    if len(datasec) == 16000:
                        self.startProcess = False
                        chunkAdded = False
                        highPitch = False

                        self.writeWav("audios/check.wav", b''.join(tobeCheck), sample_size)
                        sample, sample_rate = librosa.load("audios/check.wav", sr=16000)
                        predictData = librosa.resample(sample, 16000, 8000)

                        label, maxProb, prob = speechModel.predictWord(sample)
                        print("Prediction ----")
                        print("Label: " + str(label) + " Probability: " + str(maxProb))

                        if label == 'marvin' and maxProb > 0.18:
                            print("Recording now....")
                            recordNow = True
                            mChat.justPlay("audios/bell.wav")

                        tobeCheck = []
                        tobeSent = []
                        datasec = []
                        self.startProcess = True
                else:
                    tobeSent.append(bytesData)

                    if len(tobeCheck) == 60:
                        self.startProcess = False
                        print("Answering now....")
                        chunkAdded = False
                        highPitch = False

                        self.writeWav("audios/send.wav", b''.join(tobeSent), sample_size)
                        recordNow = False
                        tobeCheck = []
                        tobeSent = []
                        datasec = []

                        text = mChat.speechToTextGoogle(self.getBase64("audios/send.wav"), "wav", lang)
                        text = text.lower()

                        print("Speech to text")
                        print(text)

                        if "language" in text or "اللغه" in text:
                            if "english" in text or "الانجليزيه" in text:
                                lang = "en"
                                text = "Language is changed to English now"
                                text_ar = "تم تغيير اللغة إلى الإنجليزية الآن"
                                text = text_ar if lang == "ar" else text

                                mChat.textToSpeechGoogle(text, lang)
                            elif "arabic" in text or "العربية" in text:
                                lang = "ar"
                                text = "Language is changed to Arabic now"
                                text_ar = "تم تغيير اللغة إلى العربية الآن"
                                text = text_ar if lang == "ar" else text

                                mChat.textToSpeechGoogle(text, lang)
                        elif ("stop" in text and "now" in text) or "توقف" in text:
                            stopNow = True
                            terminate = True
                        else:
                            print("Talking to Marvin")
                            mChat.replyToUserLocal(text, lang)

                        self.startProcess = True

            data_chunk = data
            bytes_chunk = bytesData

        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    startMarvin = True
    while not terminate:
        try:
            if startMarvin:
                startMarvin = False
                app = Application()
                app.run()
        except Exception as e:
            startMarvin = True
            print("Error occurred:", e)
