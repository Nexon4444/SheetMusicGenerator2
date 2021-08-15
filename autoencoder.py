import tensorflow as tf
import numpy as np
from music21 import stream, instrument, note, chord


class MusicAutoencoder(tf.keras.Model):

    def __init__(self, latentDim, trainChords, sequenceLength, intToChord, intToDuration):
        self.nSamples = trainChords.shape[0]
        self.nChords = trainChords.shape[1]
        self.inputDim = self.nChords * sequenceLength
        self.sequenceLength = sequenceLength
        self.model = self.autoencoder(self.inputDim, latentDim)
        self.latentDim = latentDim
        self.intToChord = intToChord
        self.intToDuration = intToDuration
        self.encoder = None
        self.decoder = None

    def autoencoder(self, inputDim, latentDim):
        # Define encoder input shape
        encoderInput = tf.keras.layers.Input(shape = (inputDim))

        # Define decoder input shape
        latent = tf.keras.layers.Input(shape =(latentDim))

        # Define dense encoding layer connecting input to latent vector
        encoded = tf.keras.layers.Dense(latentDim, activation = 'tanh')(encoderInput)

        # Define dense decoding layer connecting latent vector to output
        decoded = tf.keras.layers.Dense(inputDim, activation = 'sigmoid')(latent)

        # Define the encoder and decoder models
        self.encoder = tf.keras.Model(encoderInput, encoded)
        self.decoder = tf.keras.Model(latent, decoded)

        # Define autoencoder model
        autoencoder = tf.keras.Model(encoderInput, self.decoder(encoded))
        return autoencoder

    def train(self ):
        # Define number of samples, chords and notes, and input dimension


        def flatten_train_vector(trainChords, nSamples, inputDim):

            # Set number of latent features
            latentDim = 2

            # Convert to one-hot encoding and swap chord and sequence dimensions
            trainChords = tf.keras.utils.to_categorical(trainChords).transpose(0, 2, 1)

            # Convert data to numpy array of type float
            trainChords = np.array(trainChords, np.float)

            # Flatten sequence of chords into single dimension
            trainChordsFlat = trainChords.reshape(nSamples, inputDim)
            return trainChordsFlat

        trainChordsFlat = flatten_train_vector(self.trainChords, self.nSamples, self.inputDim)

        self.model.compile(loss='binary_crossentropy', learning_rate=0.01, optimizer='rmsprop')
        # Train autoencoder
        self.model.fit(trainChordsFlat, trainChordsFlat, epochs=500)

    def generateChords(self):
        generatedChords = self.decoder(np.random.normal(size=(1, self.latentDim))).numpy().reshape(self.nChords, self.sequenceLength).argmax(0)
        generatedStream = stream.Stream()

        generatedStream.append(instrument.Piano())
        chordSequence = [self.intToChord[c] for c in generatedChords]
        # Append notes and chords to stream object
        for j in range(len(chordSequence)):
            try:
                generatedStream.append(note.Note(chordSequence[j].replace('.', ' ')))
            except:
                generatedStream.append(chord.Chord(chordSequence[j].replace('.', ' ')))

        generatedStream.write('midi', fp=generated_dir + 'autoencoder.mid')
        return generatedChords