import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
DATAPATH= "dataset/"
sns.set(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
class Config:

    def __init__(self, size = 10000, sr = 22050, split = True, normalize = False, db = False):
        #length of the array of sampled audio
        self.size = size
        #sampling rate
        self.sr = sr
        #remove anything below 20 db
        self.split = split
        # norm?
        self.normalize = normalize
        #decibel
        self.db = db


def cut_audio(audio, step = 10000, split = True):
  audio_pieces = []
  # audio = np.array(librosa.effects.trim(audio, top_db=20))
  if split:
    y_split = librosa.effects.split(audio, top_db=20)
    for i in y_split:
      segment = audio[i[0]:i[1]]
      for j in range(0, len(segment), step):
        temp = segment[j:j+step]
        temp = librosa.util.fix_length(temp, size=step)
        audio_pieces.append(temp)
  else:
    start = 0
    end = len(audio)
    audio_pieces = []
    for i in range(start, end-step, step):
      audio_pieces.append(audio[i:i+step])
       
  return audio_pieces
  



def preprocessAudio(filePath, config = Config(), db = False):
  list_matrices = []
  y,sr = librosa.load(filePath,sr=config.sr)
  aduio_pieces = cut_audio(y, config.size, config.split)
  for audio_piece in aduio_pieces:
    if config.normalize:
      audio_piece = (audio_piece - np.mean(audio_piece)) / np.std(audio_piece)
    melspect = librosa.feature.melspectrogram(y = audio_piece)
    if config.db:
        melspect = librosa.amplitude_to_db(melspect, ref=np.max)
    list_matrices.append(melspect)
  return list_matrices

def preprocessSingleAudio(filePath, config = Config(), db = False):
  list_matrices = []
  audio_piece,sr = librosa.load(filePath,sr=config.sr)
  if config.normalize:
    audio_piece = (audio_piece - np.mean(audio_piece)) / np.std(audio_piece)
  melspect = librosa.feature.melspectrogram(y = audio_piece)
  if config.db:
      melspect = librosa.amplitude_to_db(melspect, ref=np.max)
  list_matrices.append(melspect)
  return list_matrices



def initBinaryModel(showplot = False, size = 10000, sr = 22050, epoch = 10):
  # all tracks will be the X features and classification will be the target y
  all_tracks = []
  classification = []

  # assign directory
  Train_DIR = DATAPATH+ 'training/'
  
  # iterate over files in
  # that directory
  nameIndex = []


  configs = [Config(size = size, sr = sr, split = False, normalize=False), Config(size = size, sr = sr, normalize=False)]
  i = 0
  for foldername in ["0no", "1yes"]:
    folder_path = Train_DIR+ foldername
    print(folder_path)
    # checking if it is a file
    #  if os.path.isfile(f):
    nameIndex.append(foldername)
    print("Using config: ", configs[i].__dict__)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        y,sr = librosa.load(file_path,sr=sr)
        if showplot:
          pd.Series(y).plot(figsize=(10, 5),
                      lw=1,
                      title='Norm Audio Example',
                    color=color_pal[0])
          plt.show()
        audio_piece = preprocessSingleAudio(file_path, config = configs[i])
        # for i in audio_piece:
        #    plot(i)
        all_tracks += audio_piece
        print("\t",file ,"recieved" ,len(audio_piece), "clip(s)")
        classification += ([i]*len(audio_piece))
    i += 1

  print(classification)
  print(nameIndex)


  X_train, X_test, y_train, y_test = train_test_split(np.array(all_tracks), 
                                                      np.array(classification),
                                                      test_size=0.33,
                                                      random_state=42)

  X_val, X_test, y_val, y_test = train_test_split(X_test, 
                                                  y_test,
                                                  test_size=0.5,
                                                  random_state=42)


  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
  print("size of input shape model: ",X_train.shape[1:])

  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

  X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))


  #build model

  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))


  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  # model.add(layers.Dense(1))



  model.summary()


  model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(learning_rate=0.001),
                metrics=['accuracy'])
  # X_train = X_train.reshape(1, 128, 196, 1)

  history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val))

  return model


def resultofOutput(input):
  input = [1 if prediction > 0.9 else 0 for prediction in input]
  from itertools import groupby
  calls = [key for key, group in groupby(input)]
  calls = tf.math.reduce_sum(calls).numpy()
  return input, calls

def preprocessInputData(input, config = Config()):
  input = preprocessAudio(input, config = config)
      
  input = np.array(input)
  input = np.reshape(input, (input.shape[0], input.shape[1],input.shape[2], 1))

  return input

def preprocessSingleInputData(input, config = Config()):
  input = preprocessSingleAudio(input, config = config)
      
  input = np.array(input)
  input = np.reshape(input, (input.shape[0], input.shape[1],input.shape[2], 1))

  return input
def Stats (pred_labels, true_labels):
  
  # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
  TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
  
  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
  TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
  
  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
  FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
  
  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
  FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
  
  print ('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
  print('Accuracy: ', str((TP+TN)/(TP+TN+FP+FN)*100), "%" )



def validateData(model, Config):
  VALID_PATH = DATAPATH + "validate/"



  nameIndex = []


  configs = [Config(size = size, sr = sr, split = False, normalize=False), Config(size = size, sr = sr, normalize=False)]
  i = 0
  classification = []
  output = []
  for foldername in ["0no", "1yes"]:
    
    folder_path = VALID_PATH+ foldername
    print(folder_path)
    # checking if it is a file
    #  if os.path.isfile(f):
    nameIndex.append(foldername)
    print("Using config: ", configs[0].__dict__)
    for file in os.listdir(folder_path):
      file_path = os.path.join(folder_path, file)
      print(file_path)
      audio_piece = preprocessSingleInputData(file_path, config = configs[0])

      out = model.predict(audio_piece)
      output.append(out)
      classification += ([i]*len(audio_piece))

    i += 1
  output, calls = resultofOutput(output)
  print("output  ",output)
  print("expected", classification)
  print("len of arrary:",len(output))
  Stats(np.array(output), np.array(classification))

  # model.predict()

def predictFromFile(file_path, model, Config):
    input = preprocessInputData(file_path, config = config)

    print("inputed ",file_path , ", size: " , input.shape)
    print("Using config: ", config.__dict__)

    test = model.predict(input)

    input, calls = resultofOutput(test)
    print(input)
    print("len of arrary:",len(input))
    print("# calls: ", calls)


if __name__ == '__main__':
  # i have tested some configs, and the size needs to be at least 9000 based on testing for it to fit into the model. 
  # For the sampling rate(sr), i recommend having he same sr for training and input.
  size = 10000
  sr = 22050
  model = initBinaryModel(size = size, sr = sr)
  # model = []
  config = Config(size = size, sr = sr, split = False, normalize = False)
  validateData(model, Config)
  print("\n\n")
  predictFromFile("seagull-14693.mp3", model, config)
  # file_paths = [  "hoot-46198.mp3","seagull-14693.mp3", "DariusTest.mp3"]
  # for file_path in file_paths:
  #   input = preprocessInputData(file_path, config = config)

  #   print("inputed ",file_path , ", size: " , input.shape)
  #   print("Using config: ", config.__dict__)

  #   test = model.predict(input)

  #   input, calls = resultofOutput(test)
  #   print(input)
  #   print("len of arrary:",len(input))
  #   print("# calls: ", calls)

  tf.keras.backend.clear_session()