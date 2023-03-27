import pymongo
from flask import Flask, Response, request
import base64
import json
import librosa
import librosa.display
import IPython.display as ipd

# import audioUtils
with open('mongodbKey', 'r') as file:
    MONGO_URL = file.read()
dbClient = pymongo.MongoClient(MONGO_URL)
def queryAudio(id):
    
    mydb = dbClient["mydatabase"]
    #collection
    mycol = mydb["AudiosTest"]

    myquery = { "ID": id}

    # mydoc = mycol.find(myquery)

    # for x in mydoc:
    #     print(x) 
    mydoc = mycol.find_one(myquery)
    # print(mydoc['fileBytes'])
    import io
    y, sr = librosa.load(io.BytesIO(mydoc['fileBytes']), sr=None)
    import sounddevice as sd
    sd.play(y, sr)
    sd.wait()
    return mydoc

def insertAudio(id, wavfile):
    
    mydb = dbClient["mydatabase"]
    #collection
    mycol = mydb["AudiosTest"]
    
    f = open(wavfile, "rb")
    y= f.read()
    myInsert = { "ID": id, "fileBytes" : y}

    mydoc = mycol.insert_one(myInsert)




app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def MAINROUTE():
    if(request.method == 'POST'):
        
        return " "

    else:
        # return json.dumps(out,indent = 3)
        return
    



if __name__ =='__main__':
    # print("hello world")
    # app.run(debug = True, host = '0.0.0.0')
    queryAudio("test4")
    # insertAudio("test4", "sampleaudio.wav")

