import io
from glob import glob

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pymongo



with open('mongodbKey', 'r') as file:
    MONGO_URL = file.read()
dbClient = pymongo.MongoClient(MONGO_URL)
DATABASE_NAME = "mydatabase"
COLLECTION_NAME = "AudiosTest"
def binaryData2numpy(input):
    out, sr = librosa.load(io.BytesIO(input), sr=None)
    return out
def queryAudio(id):
    mycol = dbClient[DATABASE_NAME][COLLECTION_NAME]
    myquery = { "ID": id}
    mydoc = mycol.find_one(myquery)
    return mydoc
def playNumpy(numpy_array):
    import sounddevice as sd
    sd.play(numpy_array, sr)
    sd.wait()



def insertAudio(id, wavfile):
    mycol = dbClient[DATABASE_NAME][COLLECTION_NAME]

    
    f = open(wavfile, "rb")
    y= f.read()
    myInsert = { "ID": id, "fileBytes" : y}

    mycol.insert_one(myInsert)

def generateMelSpecBinaryImage(np_array):
    # np_array, sr = librosa.load("hoot-46198.mp3", sr=22050)
    S = librosa.feature.melspectrogram(y=np_array,
                                  sr=22050,
                                  n_mels=128 * 2,)

    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    print(S_db_mel.shape)
    spectrumList = S_db_mel.tolist()
    print(spectrumList[:3])
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the mel spectogram
    img = librosa.display.specshow(S_db_mel,
                                x_axis='time',
                                y_axis='log',
                                ax=ax)
    ax.set_title('Mel Spectogram', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    data = buf.getvalue()
    buf.close()
    return data


def loadMelSpecBinary2Image(binaryImg):
    from PIL import Image

    image = Image.open(io.BytesIO(binaryImg))
    return image
def updateAudio(id, newVal):
    mycol = dbClient[DATABASE_NAME][COLLECTION_NAME]


    filter = { 'ID': id }
 
    # Values to be updated.
    newvalues = { "$set": newVal }
    
    # Using update_one() method for single
    # updation.
    mycol.update_one(filter, newvalues)

def listAudio():
    mycol = dbClient[DATABASE_NAME][COLLECTION_NAME]
    return mycol.distinct("ID")
def generateID():
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    print(timestamp)
    return timestamp
if __name__ =='__main__':
    size = 10000
    sr = 22050
    ID = "test4"
    # generateID()
    # print(listAudio())


    ### queryTestAudio
    doc = queryAudio(ID)
    print(doc['MLData'])
    audioNumpy = binaryData2numpy(doc['fileBytes'])
    Img = loadMelSpecBinary2Image(doc['AudioData']['MelSpectrumImgBytes'])
    Img.show()

    # playNumpy(audioNumpy)

    
    # #generate Image Mel Spec
    # binaryImg = generateMelSpecBinaryImage(audioNumpy)

    # ##  updateAudio
    # newVal = {"MLData":{'output': [0, 1, 1, 1, 1, 1], 'Calls': 1}, "AudioData":{'sr': sr, 'Size':size, 'clipLength': size/sr, 'MelSpectrumImgBytes': binaryImg}}
    # updateAudio(ID, newVal)


    ### InsertAudio via wav file
    # insertAudio(ID, "sampleaudio.wav")

