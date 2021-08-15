import glob
import os
import pickle

from music21 import converter, pitch, interval, instrument, note, chord
import tensorflow as tf
# Define save directory
from music21.key import Key
import numpy as np
from pathlib import Path

midi_dir = './midi_songs/'
try:
  import google.colab
  IS_ON_GOOGLE_COLAB = True
except:
  IS_ON_GOOGLE_COLAB = False

TEST_RUN = False
NORMALIZE_NOTES = True
NORMALIZATION_BOUNDARIES = [3, 4]

EPOCHS = 50
FOLDER_ROOT = "."
MODEL_DIR_PATH = FOLDER_ROOT + "/generated_models/"
MODEL_NAME = FOLDER_ROOT + "/cpc"
OCCURENCES = FOLDER_ROOT + "/data/occurences"
DATA_NOTES_DIR = FOLDER_ROOT + "/data/notes"
DATA_DURATIONS_DIR = FOLDER_ROOT + "/data/durations"
MIDI_SONGS_DIR = FOLDER_ROOT + "/midi_songs_smaller"
MIDI_GENERATED_DIR = FOLDER_ROOT + "/midi_generated"
MIDI_SONGS_REGEX = MIDI_SONGS_DIR + "/*.mid"
CHECKPOINTS = FOLDER_ROOT + "/checkpoints/"
LOGS = FOLDER_ROOT + "/logs/"

all_paths = [MODEL_DIR_PATH, MODEL_NAME, OCCURENCES, DATA_NOTES_DIR, DATA_DURATIONS_DIR, MIDI_GENERATED_DIR, CHECKPOINTS, LOGS]
for path in all_paths:
    Path(path).mkdir(parents=True, exist_ok=True)

def create_train_data():
# Create empty list for scores
    originalScores = []

    # Load and make list of stream objects
    for song in glob.glob(MIDI_SONGS_REGEX):
        score = converter.parse(song)
        originalScores.append(score)

    # Define empty lists of lists
    originalChords = [[] for _ in originalScores]
    originalDurations = [[] for _ in originalScores]
    originalKeys = []

    def transpose_amount(score):
        return -int(score.chordify().analyze('key').tonic.ps % 12)

    def monophonic(stream):
        try:
            length = len(instrument.partitionByInstrument(stream).parts)
        except:
            length = 0
        return length == 1
    # Extract notes, chords, durations, and keys


    originalScores = [song.chordify() for song in originalScores]

    for i, song in enumerate(originalScores):
        originalKeys.append(str(song.analyze('key')))
        # song.transpose
        transp_int = transpose_amount(song)
        for element in song:
            if isinstance(element, note.Note):
                originalChords[i].append(element.pitch.transpose(transp_int))
                originalDurations[i].append(element.duration.quarterLength)
            elif isinstance(element, chord.Chord):
                originalChords[i].append('.'.join(str(n.transpose(transp_int)) for n in element.pitches))
                originalDurations[i].append(element.duration.quarterLength)
        print(str(i))

    cChords = [c for (c, k) in zip(originalChords, originalKeys) if (k == 'C major')]
    cDurations = [c for (c, k) in zip(originalDurations, originalKeys) if (k == 'C major')]
    # Map unique chords to integers
    uniqueChords = np.unique([i for s in originalChords for i in s])
    chordToInt = dict(zip(uniqueChords, list(range(0, len(uniqueChords)))))

    # Map unique durations to integers
    uniqueDurations = np.unique([i for s in originalDurations for i in s])
    durationToInt = dict(zip(uniqueDurations, list(range(0, len(uniqueDurations)))))

    # Print number of unique notes and chords
    print(len(uniqueChords))

    # Print number of unique durations
    print(len(uniqueDurations))

    intToChord = {i: c for c, i in chordToInt.items()}
    intToDuration = {i: c for c, i in durationToInt.items()}

    # Define sequence length
    sequenceLength = 32

    # Define empty arrays for train data
    trainChords = []
    trainDurations = []

    # Construct training sequences for chords and durations
    for s in range(len(cChords)):
        chordList = [chordToInt[c] for c in cChords[s]]
        durationList = [durationToInt[d] for d in cDurations[s]]
        for i in range(len(chordList) - sequenceLength):
            trainChords.append(chordList[i:i+sequenceLength])
            trainDurations.append(durationList[i:i+sequenceLength])

    with open(DATA_NOTES_DIR, 'wb') as filepath:
        pickle.dump(chordList, filepath)

    with open(DATA_DURATIONS_DIR, 'wb') as filepath:
        pickle.dump(chordList, filepath)


    return trainChords, trainDurations, sequenceLength, intToChord, intToDuration

if __name__ == "__main__":
    create_train_data()
# Convert to one-hot encoding and swap chord and sequence dimensions
