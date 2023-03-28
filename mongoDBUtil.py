import io

import IPython.display as ipd
import librosa
import librosa.display
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
if __name__ =='__main__':
    size = 10000
    sr = 22050
    ID = "test4"

    print(listAudio())


    # ### queryTestAudio
    # doc = queryAudio(ID)
    # print(doc['MLData'])
    # audioNumpy = binaryData2numpy(doc['fileBytes'])
    # playNumpy(audioNumpy)

    # ##  updateAudio
    # newVal = {"MLData":{'output': [0, 1, 1, 1, 1, 1], 'Calls': 1}}
    # updateAudio(ID, newVal)


    ### InsertAudio via wav file
    # insertAudio(ID, "sampleaudio.wav")

