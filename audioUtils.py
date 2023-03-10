import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

sns.set(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# def cut_song(song, step = 10000):
#   start = 0
#   end = len(song)
#   song_pieces = []
#   for i in range(start, end, step):
#     song_pieces.append(song[i:i+step])
#   return song_pieces

def cut_song(song, step = 10000):
  song_pieces = []
  y_split = librosa.effects.split(song, top_db=20)
  for i in y_split:
    segment = song[i[0]:i[1]]
    for j in range(0, (len(segment)-int(step/2)), step):

      song_pieces.append(segment[j:j+step])
  return song_pieces
  



def preprocessAudio(filePath, size = 10000, db = False):
  list_matrices = []
  y,sr = librosa.load(filePath,sr=22050)
  song_pieces = cut_song(y, size)
  for song_piece in song_pieces:
    
    melspect = librosa.feature.melspectrogram(y = song_piece)
    if db:
        melspect = librosa.amplitude_to_db(melspect, ref=np.max)
    list_matrices.append(melspect)
  return list_matrices



if __name__ == '__main__':
  preprocessAudio("sampleaudio.wav")
  x = 0

  