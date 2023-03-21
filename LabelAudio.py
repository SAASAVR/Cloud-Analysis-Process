
# select the folder
# split the data with preprcess
# play sound for user
# sort the data with 0, 1, or something else if confused
import os

import librosa
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import audioUtils as utils
import soundfile as sf

import shutil
TRAINING_DIR = "training/"
VALIDATE_DIR = "validate/"
ARCHIVE_DIR = "archiveAudio/"
LABELPATH = {
  "y": "1yes/",
  "n": "0no/",
  "i": "2notSure/",
  "0": "notLabeled/"
}
VisualSelection = " _______     _______     _______\n"\
                  "|       |   |       |   |       |\n"\
                  "|  Yes  |   |  No   |   |  Idk  |\n"\
                  "|___y___|   |___n___|   |___i___|"


# Clear the terminal window
def clearTerminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def getUserInput():
    while True:
        user_input = input("Please enter 'y' for Yes, 'n' for No, or 'i' for Idk(e): ")
        
        if user_input.lower() not in ('y', 'n', 'i', 'e'):
            print("Invalid input. Please enter 'y', 'n', or 'i'.")
        else:
            break
            
    print("You entered:", user_input.lower())
    return user_input.lower()

def LabelAudio(folderDir, split):

    for fileName in os.listdir(folderDir + LABELPATH["0"]):
        folder_path = folderDir + LABELPATH["0"] + fileName

        y, sr = librosa.load(folder_path)
        y_split = utils.cut_audio(y, 10000, True)
        index = 0


        # Plot the sound file
        for i in y_split:
            sd.play(i, 22050,blocking=True)
            newfile = fileName.replace(".mp3", "_" + str(index) + ".wav")
            index = index + 1
            clearTerminal()
            print("playing ", newfile)
            print(VisualSelection)
            input2 = getUserInput()
            if (input2 == 'e'):
                break
            fileOutPath = folderDir+ LABELPATH[input2] + newfile
            print("writing to ", fileOutPath)
            sf.write(fileOutPath, i, sr, 'PCM_24')
        shutil.move(folder_path, ARCHIVE_DIR + fileName)
        print("Moved og file to ", ARCHIVE_DIR )
            # librosa.write_wav('training/' + newfile, i, sr)

if __name__ == '__main__':
    while True:
        user_input = input("Please enter to 't' or 'v' to label training or validation: ")
        
        if user_input.lower() not in ('t', 'v'):
            print("Invalid input. Please enter to 't' or 'v' to label training or validation: ")
        else:
            break
    if (user_input.lower() == "t"):
        dir = TRAINING_DIR
    else:
        dir = VALIDATE_DIR
    while True:
        user_input = input("Please enter 'y' for Yes, 'n' for No to filter data files: ")
        
        if user_input.lower() not in ('y', 'n'):
            print("Invalid input. Please enter 'y' for Yes, 'n' for No to filter data files: ")
        else:
            break
    if (user_input.lower() == "y"):
        split = True
    else:
        split = False
    
    LabelAudio(dir, split)

