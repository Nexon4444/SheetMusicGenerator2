# %%

import tensorflow as tf
import numpy as np
from music21 import stream, instrument, note, note
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint

try:
    import google.colab

    IS_ON_GOOGLE_COLAB = True
except:
    IS_ON_GOOGLE_COLAB = False

if IS_ON_GOOGLE_COLAB:
    from google.colab import drive

    drive.mount('/content/drive')

# %%
#
# print(tf.test.is_gpu_available())
# tf.config.list_physical_devices('GPU')

# %%

import glob
import os
import pickle

from music21 import converter, pitch, interval, instrument, chord, note
import tensorflow as tf
# Define save directory
from music21.key import Key
import numpy as np
from pathlib import Path

midi_dir = './midi_songs/'


def get_current_datetime():
    from datetime import datetime
    now = datetime.now()
    dt_name = now.strftime("%m_%d_%Y__%H_%M_%S")
    return dt_name


if IS_ON_GOOGLE_COLAB:
    FOLDER_ROOT = os.path.join("content", "drive", "MyDrive", "magisterka", "SheetMusicGenerator2")
else:
    FOLDER_ROOT = os.path.join(".")

TEST_RUN = False
NORMALIZE_NOTES = True
NORMALIZATION_BOUNDARIES = [3, 4]

EPOCHS = 10
BATCH_SIZE = 512
AUTOENCODER = "AUTOENCODER"
MODEL_NAME = AUTOENCODER

MODEL_FOLDER_ROOT = os.path.join(FOLDER_ROOT, MODEL_NAME)
CURR_DT = get_current_datetime()
MODEL_DIR_PATH = os.path.join(MODEL_FOLDER_ROOT, "generated_models")
OCCURENCES = os.path.join(MODEL_FOLDER_ROOT, "data", "occurences")
DATA_DIR = os.path.join(MODEL_FOLDER_ROOT, "data")

DATA_NOTES_DIR = os.path.join(DATA_DIR, "notes")
DATA_DURATIONS_DIR = os.path.join(DATA_DIR, "durations")
DATA_DICTS_DIR = os.path.join(DATA_DIR, "dicts")

DATA_INT_TO_NOTE_PATH = os.path.join(DATA_DICTS_DIR, "int_to_note_" + str(CURR_DT))
DATA_INT_TO_DURATION_PATH = os.path.join(DATA_DICTS_DIR, "int_to_duration_" + str(CURR_DT))
DATA_NOTES_PATH = os.path.join(DATA_NOTES_DIR, "notes_" + str(CURR_DT))
DATA_DURATIONS_PATH = os.path.join(DATA_DURATIONS_DIR, "durations_" + str(CURR_DT))

# MIDI_SONGS_DIR = os.path.join(FOLDER_ROOT, "midi_songs")
MIDI_SONGS_DIR = os.path.join(FOLDER_ROOT, "midi_songs_smaller")
MIDI_GENERATED_DIR = os.path.join(MODEL_FOLDER_ROOT, "midi_generated")
MIDI_SONGS_REGEX = os.path.join(MIDI_SONGS_DIR, "*.mid")
CHECKPOINTS_DIR = os.path.join(MODEL_FOLDER_ROOT, "checkpoints")
CHECKPOINT = os.path.join(CHECKPOINTS_DIR, str(CURR_DT))
LOGS_DIR = os.path.join(MODEL_FOLDER_ROOT, "logs")
LOG = os.path.join(LOGS_DIR, str(CURR_DT))

COMPUTED_INT_TO_NOTE_PATH = "C:\\Users\\Nexon\\PycharmProjects\\SheetMusicGenerator2\\AUTOENCODER\\data\\dicts\\int_to_note_08_16_2021__20_10_21"
COMPUTED_INT_TO_DURATION_PATH = "C:\\Users\\Nexon\\PycharmProjects\\SheetMusicGenerator2\\AUTOENCODER\\data\\dicts\\int_to_duration_08_16_2021__20_10_21"
COMPUTED_NOTES_PATH = "C:\\Users\\Nexon\\PycharmProjects\\SheetMusicGenerator2\\AUTOENCODER\\data\\notes\\notes_08_16_2021__20_10_21"
COMPUTED_DURATIONS_PATH = "C:\\Users\\Nexon\\PycharmProjects\\SheetMusicGenerator2\\AUTOENCODER\\data\\durations\\durations_08_16_2021__20_10_21"

all_paths = [MODEL_DIR_PATH, MODEL_NAME, OCCURENCES, DATA_NOTES_DIR, DATA_DURATIONS_DIR, DATA_DICTS_DIR,
             MIDI_GENERATED_DIR, CHECKPOINTS_DIR, CHECKPOINT, LOGS_DIR, LOG]
for path in all_paths:
    Path(path).mkdir(parents=True, exist_ok=True)


# if __name__ == "__main__":
#     create_train_data()
# # Convert to one-hot encoding and swap note and sequence dimensions

# %%

class MusicAutoencoder():
    def __init__(self, latent_dim, sequence_length, train_notes_path=None, train_durations_path=None,
                 int_to_note_path=None, int_to_duration_path=None):
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.tensor_dataset = None
        self.input_dim = None
        self.n_notes = None
        self.encoder = None
        self.decoder = None
        self.numpy_dataset = None

        self.train_notes_path = train_notes_path
        self.train_durations_path = train_durations_path
        self.int_to_note_path = int_to_note_path
        self.int_to_duration_path = int_to_duration_path

        if train_notes_path is None or train_durations_path is None or int_to_note_path is None or int_to_duration_path is None:
            self.parse_songs()

        else:
            with open(train_notes_path, 'rb') as train_notes_file:
                self.train_notes = pickle.load(train_notes_file)

            with open(train_notes_path, 'rb') as train_durations_file:
                self.train_durations = pickle.load(train_durations_file)

            with open(train_notes_path, 'rb') as int_to_note_file:
                self.int_to_note = pickle.load(int_to_note_file)

            with open(train_notes_path, 'rb') as int_to_duration_file:
                self.int_to_duration = pickle.load(int_to_duration_file)


        self.prepare_data()
        self.model = self.autoencoder()
        self.steps_per_epoch = len(self.train_notes) // BATCH_SIZE

    # def create_autoencoder(self):
    #         self.model = self.autoencoder(self.input_dim, self.latent_dim)

    def autoencoder(self):
        # Define encoder input shape
        encoder_input = tf.keras.layers.Input(shape=self.input_dim)

        # Define decoder input shape
        latent = tf.keras.layers.Input(shape=self.latent_dim)

        # Define dense encoding layer connecting input to latent vector
        encoded = tf.keras.layers.Dense(self.latent_dim, activation='tanh')(encoder_input)

        # Define dense decoding layer connecting latent vector to output
        decoded = tf.keras.layers.Dense(self.input_dim, activation='sigmoid')(latent)

        # Define the encoder and decoder models
        self.encoder = tf.keras.Model(encoder_input, encoded)
        self.decoder = tf.keras.Model(latent, decoded)

        # Define autoencoder model
        autoencoder = tf.keras.Model(encoder_input, self.decoder(encoded))
        return autoencoder

    def data_generator(self):
        """Replaces Keras' native ImageDataGenerator."""
        # i = 0
        # file_list = os.listdir(directory)
        # dataset = tf.data.Dataset.from_tensors(self.numpy_dataset).repeat(EPOCHS)
        yielded_size = 0
        for i in range(EPOCHS):
            for batch in np.array_split(self.numpy_dataset, self.steps_per_epoch):
                yielded_size += len(batch)
                print(f" yielded_size: {yielded_size}")
                yield batch, batch

    # def data_generator(self):
    #
    #     # i = 0
    #     # file_list = os.listdir(directory)
    #     with open()
    #     for batch in self.tensor_dataset.batch(BATCH_SIZE):
    #         print(batch)
    #         yield batch
    #     # while True:
    #     batch = []
    #     for b in range(batch_size):
    #         self.tensor_dataset.batch(BATCH_SIZE)
    #         if i == len(file_list):
    #             i = 0
    #         sample = file_list[i]
    #         i += 1
    #         # image = cv2.resize(cv2.imread(sample[0]), INPUT_SHAPE)
    #         # image_batch.append((image.astype(float) - 128) / 128)
    #
    #     yield np.array(image_batch)

    def train(self, checkpoint_path=None):
        # Define number of samples, notes and notes, and input dimension
        # filepath = CHECKPOINTS + "weights-improvement-{epoch:02d}-{loss:.4f}-{categorical_accuracy:.4f}-bigger.hdf5"
        # filepath = "weights-improvement-epoch:{epoch:02d}-loss:{loss:.4f}-cat_acc:{categorical_accuracy:.4f}.hdf5"

        if checkpoint_path:
            self.model.load_weights(checkpoint_path)

        # filepath = os.path.join(CHECKPOINT, "epoch={epoch:02d}-loss={loss:.4f}-acc={binary_accuracy:.4f}.hdf5")
        filepath = os.path.join(CHECKPOINT, "epoch={epoch:02d}-loss={loss:.4f}-acc={categorical_accuracy:.4f}.hdf5")

        # filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='categorical_accuracy',
            verbose=0,
            save_best_only=True,
            mode='max'
        )
        log = tf.keras.callbacks.TensorBoard(log_dir=LOG),

        callbacks_list = [checkpoint, log]
        # history = self.model.fit(network_input, network_output, epochs=EPOCHS, batch_size=128, callbacks=callbacks_list)
        # model.save(MODEL_DIR_PATH + MODEL_NAME + "_" + CURR_DT + ".hdf5")

        # self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy()])
        self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.01),
                           metrics=[tf.keras.metrics.CategoricalAccuracy()])
        # self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
        # Train autoencoder
        self.model.summary()
        print(MODEL_DIR_PATH + MODEL_NAME + "_" + CURR_DT + ".hdf5")
        # history = self.model.fit(self.trainNotesFlat, self.trainNotesFlat, epochs=1)
        # history = self.model.fit(self.trainNotesFlat, self.trainNotesFlat, epochs=500, callbacks=callbacks_list, batch_size=8)
        # tensor_dataset = tf.data.Dataset.from_tensors((self.trainNotesFlat, self.trainNotesFlat))

        history = self.model.fit(x=self.data_generator(), epochs=EPOCHS, callbacks=callbacks_list,
                                 batch_size=BATCH_SIZE, steps_per_epoch=self.steps_per_epoch)

        print(f"steps_per_epoch: {self.steps_per_epoch}")
        print(f"BATCH_SIZE: {BATCH_SIZE}")
        print(f"len(self.train_notes): {len(self.train_notes)}")
        print(history.history)
        print(MODEL_DIR_PATH + MODEL_NAME + "_" + CURR_DT + ".hdf5")
        self.model.save(os.path.join(MODEL_DIR_PATH, MODEL_NAME + "_" + CURR_DT + ".hdf5"))

    def generate_notes(self):
        generated_notes = self.decoder(np.random.normal(size=(1, self.latent_dim))) \
            .numpy().reshape(self.n_notes, self.sequence_length) \
            .argmax(0)

        generated_stream = stream.Stream()
        generated_stream.append(instrument.Piano())
        note_sequence = [self.int_to_note[c] for c in generated_notes]
        # Append notes and notes to stream object
        for j in range(len(note_sequence)):
            try:
                generated_stream.append(note.Note(note_sequence[j].replace('.', ' ')))
            except:
                generated_stream.append(note.Note(note_sequence[j].replace('.', ' ')))

        generated_stream.write('midi', fp=MIDI_GENERATED_DIR + 'autoencoder.mid')
        # return generatedNotes

    def parse_songs(self):
        # Create empty list for scores
        original_scores = []

        # Load and make list of stream objects
        for song in glob.glob(MIDI_SONGS_REGEX):
            print("Parsing song: " + str(song))
            score = converter.parse(song)
            original_scores.append(score)

        # Define empty lists of lists
        original_notes = [[] for _ in original_scores]
        original_durations = [[] for _ in original_scores]
        original_keys = []

        def transpose_amount(score):
            return -int(score.chordify().analyze('key').tonic.ps % 12)

        def monophonic(stream):
            try:
                length = len(instrument.partitionByInstrument(stream).parts)
            except:
                length = 0
            return length == 1

        # Extract notes, notes, durations, and keys

        original_scores = [song.chordify() for song in original_scores]

        for i, song in enumerate(original_scores):

            # song.transpose
            transp_int = transpose_amount(song)
            original_keys.append(str(song.analyze('key').transpose(transp_int)))

            for element in song:
                if isinstance(element, note.Note):
                    original_notes[i].append(element.pitch.transpose(transp_int))
                    original_durations[i].append(element.duration.quarterLength)

                elif isinstance(element, chord.Chord):
                    original_notes[i].append('.'.join(str(n.transpose(transp_int)) for n in element.pitches))
                    original_durations[i].append(element.duration.quarterLength)

            print(str(original_keys[i]))

        c_notes = [c for (c, k) in zip(original_notes, original_keys) if (k == 'C major')]

        c_durations = [c for (c, k) in zip(original_durations, original_keys) if (k == 'C major')]
        # Map unique notes to integers
        unique_notes = np.unique([i for s in original_notes for i in s])
        note_to_int = dict(zip(unique_notes, list(range(0, len(unique_notes)))))

        # Map unique durations to integers
        unique_durations = np.unique([i for s in original_durations for i in s])
        duration_to_int = dict(zip(unique_durations, list(range(0, len(unique_durations)))))

        # Print number of unique notes and notes
        print(len(unique_notes))

        # Print number of unique durations
        print(len(unique_durations))

        int_to_note = {i: c for c, i in note_to_int.items()}
        int_to_duration = {i: c for c, i in duration_to_int.items()}

        # Define sequence length

        # Define empty arrays for train data
        train_notes = []
        train_durations = []

        # Construct training sequences for notes and durations
        for s in range(len(c_notes)):
            note_list = [note_to_int[c] for c in c_notes[s]]
            duration_list = [duration_to_int[d] for d in c_durations[s]]
            for i in range(len(note_list) - self.sequence_length):
                train_notes.append(note_list[i:i + self.sequence_length])
                train_durations.append(duration_list[i:i + self.sequence_length])

        with open(DATA_NOTES_PATH, 'wb') as filepath:
            pickle.dump(train_notes, filepath)

        with open(DATA_DURATIONS_PATH, 'wb') as filepath:
            pickle.dump(train_durations, filepath)

        with open(DATA_INT_TO_NOTE_PATH, 'wb') as filepath:
            pickle.dump(int_to_note, filepath)

        with open(DATA_INT_TO_DURATION_PATH, 'wb') as filepath:
            pickle.dump(int_to_duration, filepath)

        self.train_notes = train_notes
        self.train_durations = train_durations
        self.int_to_note = int_to_note
        self.int_to_duration = int_to_duration

    def prepare_data(self):
        # print("trainNotesFlat: " + str(train_notes))
        train_notes_categorical = tf.keras.utils.to_categorical(self.train_notes, dtype="float16").transpose(0, 2, 1)
        # Convert data to numpy array of type float
        # trainNotes = np.array(trainNotes, np.float32)

        n_samples = train_notes_categorical.shape[0]
        self.n_notes = train_notes_categorical.shape[1]
        self.input_dim = self.n_notes * self.sequence_length
        # Flatten sequence of notes into single dimension
        self.numpy_dataset = train_notes_categorical.reshape(n_samples, self.input_dim)
        # self.numpy_dataset = train_notes_categorical
        # self.tensor_dataset = tf.data.Dataset.from_tensors(tensors=(train_notes_flattened, train_notes_flattened))

        # return tensor_dataset, input_dim, train_durations, sequence_length, int_to_note, int_to_duration, n_notes


# %%

class ModelFactory:
    def factory(self, model_type, use_computed_values):
        if model_type == AUTOENCODER:
            if use_computed_values:
                model = MusicAutoencoder(2, 32, COMPUTED_NOTES_PATH, COMPUTED_DURATIONS_PATH, COMPUTED_INT_TO_NOTE_PATH,
                                         COMPUTED_INT_TO_DURATION_PATH)
            else:
                model = MusicAutoencoder(2, 32)
            return model


# %%

modelFactory = ModelFactory()
music_autoencoder = modelFactory.factory(MODEL_NAME, True)
music_autoencoder.train()
music_autoencoder.generate_notes()
