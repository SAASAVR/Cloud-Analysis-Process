import pymongo
from flask import Flask, Response, request
import base64
import json

import audioUtils

MONGO_URL = ""
def queryAudio(id):
    dbClient = pymongo.MongoClient(MONGO_URL)
    mydb = dbClient["mydatabase"]
    #collection
    mycol = mydb["Audios"]

    myquery = { "ID": id}

    mydoc = mycol.find(myquery)

    for x in mydoc:
        print(x) 
    return mydoc

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def MAINROUTE():
    if(request.method == 'POST'):
        
        return " "

    else:
        # return json.dumps(out,indent = 3)
        return
    



if __name__ =='__main__':
    print("hello world")
    app.run(debug = True, host = '0.0.0.0')

