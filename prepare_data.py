import os
from music21 import converter, pitch, interval, instrument, note, chord
import tensorflow as tf
# Define save directory
from music21.key import Key
import numpy as np

midi_dir = './midi_songs/'

# Identify list of MIDI files
songList = os.listdir(midi_dir)

# Create empty list for scores
originalScores = []

# Load and make list of stream objects
for song in songList:
    score = converter.parse(midi_dir+song)
    originalScores.append(score)

# Define empty lists of lists
originalChords = [[] for _ in originalScores]
originalDurations = [[] for _ in originalScores]
originalKeys = []

def transpose_amount(key: Key):
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

# Convert to one-hot encoding and swap chord and sequence dimensions
trainChords = tf.keras.utils.to_categorical(trainChords).transpose(0, 2, 1)

# Convert data to numpy array of type float
trainChords = np.array(trainChords, np.float)

# Flatten sequence of chords into single dimension
trainChordsFlat = trainChords.reshape(nSamples, inputDim)