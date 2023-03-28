import IPython.display as ipd
import librosa
import librosa.display
import pymongo
from flask import Flask, Response, request

from audioUtils import *
from mongoDBUtil import *

# import audioUtils


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def MAINROUTE():
    if(request.method == 'POST'):
        
        return " "

    else:
        # return json.dumps(out,indent = 3)
        return
    



if __name__ =='__main__':
    size = 10000
    sr = 22050
    ID = "test4"
    # print("hello world")
    # app.run(debug = True, host = '0.0.0.0')

    ### queryTestAudio
    doc = queryAudio(ID)
    print(doc['MLData'])


    ### queryAudioThrouhgML
    # model = initBinaryModel(size = size, sr = sr)
    # # model = []
    # config = Config(size = size, sr = sr, split = False, normalize = False)
    # input = queryAudio(ID)
    # y, sr = librosa.load(io.BytesIO(input['fileBytes']), sr=None)
    # output, calls = predictFromArrayList(y, model, config)
    # newVal = {"MLData":{"output" :output, "Calls": calls}}
    # updateAudio(ID, newVal)


    ### InsertAudio via wav file
    # insertAudio("test4", "sampleaudio.wav")

