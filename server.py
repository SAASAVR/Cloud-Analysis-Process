import IPython.display as ipd
import librosa
import librosa.display
import pymongo
from flask import Flask, Response, request
import json
from audioUtils import *
from mongoDBUtil import *
from playsound import playsound
 
# import audioUtils


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def MAINROUTE():
    global model, config
    if(request.method == 'POST'):
        request_data = json.loads(request.data.decode('utf-8'))
        print(request_data['ID'])

        input = queryAudio(ID)
        y, sr = librosa.load(io.BytesIO(input['fileBytes']), sr=None)
        output, calls = predictFromArrayList(y, model, config)
        newVal = {"MLData":{"output" :output, "Calls": calls}}
        updateAudio(ID, newVal)
        playsound("sound/NewnotificationUNIVERSFEILD.mp3")
            
        return json.dumps({"success":True})

    else:
        return " "
    


if __name__ =='__main__':
    size = 10000
    sr = 22050
    ID = "2023-03-31_121144"
    # ID = "31_121144"
    # ## queryAudioThrouhgML
    model = initBinaryModel(size = size, sr = sr)
    # model = []
    config = Config(size = size, sr = sr, split = False, normalize = False)
    playsound("sound/NewnotificationUNIVERSFEILD.mp3")


    print("hello world")
    app.run(debug = True, host = "0.0.0.0")
    
    # ### queryTestAudio
    # doc = queryAudio(ID)
    # print(doc['MLData'])


    # input = queryAudio(ID)
    # y, sr = librosa.load(io.BytesIO(input['fileBytes']), sr=sr)
    # import sounddevice as sd
    # print(sr)
    # # sd.play(y, sr)
    # # sd.wait()
    # print(y.shape)
    # output, calls = predictFromArrayList(y, model, config)
    # # binaryImg = generateMelSpecBinaryImage(y)
    # # newVal = {"MLData":{"output" :output, "Calls": 3}}
    # newVal = {"MLData":{'output': output, 'Calls': calls}}

    # playsound("sound/NewnotificationUNIVERSFEILD.mp3")

    # # "AudioData":{'sr': sr, 'Size':size, 'clipLength': size/sr, 'MelSpectrumImgBytes': binaryImg}
    # # updateAudio(ID, newVal)


    # ### InsertAudio via wav file
    # # insertAudio("test4", "sampleaudio.wav")
    # wavfile = "Archive/Owl/dataset/archiveAudio/owl_hooting_000102_0145s3-002-070-000-002-068-074-37397.mp3"
    # # insertAudio(generateID(), wavfile)
    # insertAudio("test30", wavfile, sr, size)

