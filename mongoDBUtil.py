import io
from glob import glob

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
"""This is the query audio when user selects one of the audio links"""
def queryAudio(id):
    mycol = dbClient[DATABASE_NAME][COLLECTION_NAME]
    myquery = { "ID": id}
    mydoc = mycol.find_one(myquery)
    return mydoc
"""you might need this if you would like to listen to it"""
"""Takes in numpy float array of librosa, plays sound"""
def playNumpy(numpy_array, sr):
    import sounddevice as sd
    sd.play(numpy_array, sr)
    sd.wait()

"""Convert the 'Bytefile to a numpy float"""
def binaryData2numpy(input):
    out, sr = librosa.load(io.BytesIO(input), sr=None)
    return out

"""changes the numpy array to a mel spectrum image in binary"""
def generateMelSpecBinaryImage(np_array):
    # np_array, sr = librosa.load("hoot-46198.mp3", sr=22050)
    S = librosa.feature.melspectrogram(y=np_array,
                                  sr=22050,
                                  n_mels=128 * 2,)

    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    spectrumList = S_db_mel.tolist()
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
"""Initial insert of audio to database, contains ID, the audio data in binary, sampling rate(sr), an image of mel spectrum"""
def insertAudio(id, wavfile, sr, size = 10000):
    mycol = dbClient[DATABASE_NAME][COLLECTION_NAME]
    f = open(wavfile, "rb")
    y= f.read()
    binaryImg = generateMelSpecBinaryImage(binaryData2numpy(y))
    myInsert = {"ID": id, "fileBytes" : y, "AudioData":{'sr': sr, 'Size':size, 'clipLength': size/sr, 'MelSpectrumImgBytes': binaryImg}, "MLData":{}}
    mycol.insert_one(myInsert)
"""leads binary image to numpy or 2D format"""
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
"""query the list of audios by unique ID"""
def listAudio():
    mycol = dbClient[DATABASE_NAME][COLLECTION_NAME]
    return mycol.distinct("ID")
"""generates ID for it to be send to db and for querying"""
def generateID():
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    print(timestamp)
    return timestamp
"""loads the binary image to an actual redable html image"""
def image2HtmlSrc(binaryBuffer):
    img_str = base64.b64encode(binaryBuffer).decode('utf-8')
    img_str = "data:image/png;base64, " + img_str
    return img_str
if __name__ =='__main__':
    size = 10000
    sr = 22050
    # ID = "2023-03-31_121144"
    ID = generateID()
    # # print(listAudio())


    # ### queryTestAudio
    # doc = queryAudio(ID)
    # audioNumpy = binaryData2numpy(doc['fileBytes'])
    # # Img = loadMelSpecBinary2Image(doc['AudioData']['MelSpectrumImgBytes'])
    # import base64
    # # buffer = doc['AudioData']['MelSpectrumImgBytes']
    
    # # Img.show()

    # # playNumpy(audioNumpy)

    
    # # #generate Image Mel Spec
    # binaryImg = generateMelSpecBinaryImage(audioNumpy)

    # # ##  updateAudio
    # # newVal = {"AudioData":{'sr': sr, 'Size':size, 'clipLength': size/sr, 'MelSpectrumImgBytes': binaryImg}}
    # newVal = {"MLData":{}}
    # updateAudio(ID, newVal)

    file = "dataset/training/0no/fidgetToy_s2.wav"
    ## InsertAudio via wav file
    insertAudio(ID, file, sr, size)

