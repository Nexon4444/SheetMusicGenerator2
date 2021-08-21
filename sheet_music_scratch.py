# %%


import tensorflow as tf
import numpy as np
from music21 import stream, instrument, note, chord
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import random

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle

from music21 import converter, pitch, interval, instrument, note, note
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
    FOLDER_ROOT = os.path.join("drive", "MyDrive", "magisterka", "SheetMusicGenerator2")
else:
    FOLDER_ROOT = os.path.join(".")

TEST_RUN = False
NORMALIZE_NOTES = True
NORMALIZATION_BOUNDARIES = [3, 4]

EPOCHS = 100
BATCH_SIZE = 256
FILTERED_KEYS = ''
TRAIN_SIZE = 0.8
USE_COMPUTED_VALUES = False

COMPUTED_INT_TO_NOTE_PATH = "AUTOENCODER/data/dicts/int_to_note_08_20_2021__19_38_03"
COMPUTED_INT_TO_DURATION_PATH = "AUTOENCODER/data/dicts/int_to_duration_08_20_2021__19_38_03"
COMPUTED_NOTES_PATH = "AUTOENCODER/data/train/notes/notes_train08_20_2021__19_38_03"
COMPUTED_DURATIONS_PATH = "AUTOENCODER/data/train/durations/durations_train08_20_2021__19_38_03"
COMPUTED_TEST_NOTES_PATH = "AUTOENCODER/data/test/notes/notes_test08_20_2021__19_38_03"
COMPUTED_TEST_DURATIONS_PATH = "AUTOENCODER/data/test/durations/durations_test08_20_2021__19_38_03"

USE_SAVE_POINT = False
SAVE_POINT = "AUTOENCODER/checkpoints/08_19_2021__18_34_10/epoch=014-loss=383.5284-acc=0.0000.hdf5"

AUTOENCODER = "AUTOENCODER"
MODEL_NAME = AUTOENCODER

MODEL_FOLDER_ROOT = os.path.join(FOLDER_ROOT, MODEL_NAME)
CURR_DT = get_current_datetime()
MODEL_DIR_PATH = os.path.join(MODEL_FOLDER_ROOT, "generated_models")
OCCURENCES = os.path.join(MODEL_FOLDER_ROOT, "data", "occurences")
DATA_DIR = os.path.join(MODEL_FOLDER_ROOT, "data")

DATA_TRAIN_FOLDER = os.path.join(DATA_DIR, "train")
DATA_TEST_FOLDER = os.path.join(DATA_DIR, "test")

DATA_TRAIN_NOTES_DIR = os.path.join(DATA_TRAIN_FOLDER, "notes")
DATA_TRAIN_DURATIONS_DIR = os.path.join(DATA_TRAIN_FOLDER, "durations")
DATA_TEST_NOTES_DIR = os.path.join(DATA_TEST_FOLDER, "notes")
DATA_TEST_DURATIONS_DIR = os.path.join(DATA_TEST_FOLDER, "durations")
DATA_DICTS_DIR = os.path.join(DATA_DIR, "dicts")

DATA_INT_TO_NOTE_PATH = os.path.join(DATA_DICTS_DIR, "int_to_note_" + str(CURR_DT))
DATA_INT_TO_DURATION_PATH = os.path.join(DATA_DICTS_DIR, "int_to_duration_" + str(CURR_DT))
DATA_TRAIN_NOTES_PATH = os.path.join(DATA_TRAIN_NOTES_DIR, "notes_train" + str(CURR_DT))
DATA_TRAIN_DURATIONS_PATH = os.path.join(DATA_TRAIN_DURATIONS_DIR, "durations_train" + str(CURR_DT))
DATA_TEST_NOTES_PATH = os.path.join(DATA_TEST_NOTES_DIR, "notes_test" + str(CURR_DT))
DATA_TEST_DURATIONS_PATH = os.path.join(DATA_TEST_DURATIONS_DIR, "durations_test" + str(CURR_DT))

# MIDI_SONGS_DIR = os.path.join(FOLDER_ROOT, "midi_songs")
# MIDI_SONGS_DIR = os.path.join(FOLDER_ROOT, "midi_songs_smaller")
MIDI_SONGS_DIR = os.path.join(FOLDER_ROOT, "midi_songs_medium")
MIDI_GENERATED_DIR = os.path.join(MODEL_FOLDER_ROOT, "midi_generated")
MIDI_SONGS_REGEX = os.path.join(MIDI_SONGS_DIR, "*.mid")
CHECKPOINTS_DIR = os.path.join(MODEL_FOLDER_ROOT, "checkpoints")
CHECKPOINT = os.path.join(CHECKPOINTS_DIR, str(CURR_DT))
LOGS_DIR = os.path.join(MODEL_FOLDER_ROOT, "logs")
LOG = os.path.join(LOGS_DIR, str(CURR_DT))

all_paths = [MODEL_DIR_PATH, OCCURENCES, DATA_TRAIN_NOTES_DIR, DATA_TRAIN_DURATIONS_DIR, DATA_TEST_NOTES_DIR,
             DATA_TEST_DURATIONS_DIR, DATA_DICTS_DIR, DATA_DICTS_DIR,
             MIDI_GENERATED_DIR, CHECKPOINTS_DIR, CHECKPOINT, LOGS_DIR, LOG]

for path in all_paths:
    Path(path).mkdir(parents=True, exist_ok=True)


# if __name__ == "__main__":
#     create_train_data()
# # Convert to one-hot encoding and swap note and sequence dimensions

# %%

class MusicAutoencoder():
    def __init__(self, model_type, latent_dim, sequence_length, train_notes_path=None,
                 train_durations_path=None, test_notes_path=None, test_durations_path=None,
                 int_to_note_path=None, int_to_duration_path=None):
        self.model_type = model_type
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.tensor_dataset = None
        self.input_dim = None
        self.n_of_unique_notes_classes = None
        self.encoder = None
        self.decoder = None
        self.numpy_dataset = None

        self.train_notes_path = train_notes_path
        self.train_durations_path = train_durations_path
        self.int_to_note_path = int_to_note_path
        self.int_to_duration_path = int_to_duration_path
        self.int_to_note = None
        self.int_to_duration = None


        if train_notes_path is None or train_durations_path is None or test_notes_path is None or test_durations_path is None or \
                int_to_note_path is None or int_to_duration_path is None:
            self.parse_songs()

        else:
            with open(train_notes_path, 'rb') as train_notes_file:
                self.train_notes = pickle.load(train_notes_file)

            with open(train_durations_path, 'rb') as train_durations_file:
                self.train_durations = pickle.load(train_durations_file)

            with open(test_notes_path, 'rb') as test_notes_file:
                self.test_notes = pickle.load(test_notes_file)

            with open(test_durations_path, 'rb') as test_durations_file:
                self.test_durations = pickle.load(test_durations_file)

            with open(int_to_note_path, 'rb') as int_to_note_file:
                self.int_to_note = pickle.load(int_to_note_file)

            with open(int_to_duration_path, 'rb') as int_to_duration_file:
                self.int_to_duration = pickle.load(int_to_duration_file)

        self.train_dataset = self.prepare_data(self.train_notes)
        self.test_dataset = self.prepare_data(self.test_notes)
        self.model = self.autoencoder()
        self.steps_per_epoch = len(self.train_dataset) // BATCH_SIZE

    # def create_autoencoder(self):
    #         self.model = self.autoencoder(self.input_dim, self.latent_dim)

    def autoencoderNew(self):
        class Autoencoder(Model):
            def __init__(self, latent_dim, sequence_length):
                super(Autoencoder, self).__init__()
                self.latent_dim = latent_dim
                self.sequence_length = sequence_length
                self.encoder = tf.keras.Sequential([
                    Input(shape=self.sequence_length, batch_size=BATCH_SIZE),
                    Dense(latent_dim, activation='tanh')
                ])
                self.decoder = tf.keras.Sequential([
                    Input(shape=self.latent_dim),
                    Dense(self.sequence_length, activation='sigmoid')
                ])

            def call(self, x, **kwargs):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        return Autoencoder(self.latent_dim, self.sequence_length)

    def autoencoder(self):
        # Define encoder input shape
        encoder_input = Input(shape=self.input_dim)

        # Define decoder input shape
        latent = Input(shape=self.latent_dim)

        # Define dense encoding layer connecting input to latent vector
        encoded = Dense(self.latent_dim, activation='tanh')(encoder_input)

        # Define dense decoding layer connecting latent vector to output
        decoded = Dense(self.input_dim, activation='sigmoid')(latent)

        # Define the encoder and decoder models
        self.encoder = Model(encoder_input, encoded)
        self.decoder = Model(latent, decoded)

        # Define autoencoder model
        autoencoder = Model(encoder_input, self.decoder(encoded))
        return autoencoder

    def data_generator(self, dataset):
        """Replaces Keras' native ImageDataGenerator."""
        # i = 0
        # file_list = os.listdir(directory)
        # dataset = tf.data.Dataset.from_tensors(self.numpy_dataset).repeat(EPOCHS)
        # def elements2remove():
        elements2remove = len(dataset) - self.steps_per_epoch * BATCH_SIZE
        yielded_size = 0
        # while True:
        for i in range(EPOCHS):
            for batch in np.array_split(dataset[:-elements2remove], self.steps_per_epoch):
                # yielded_size += len(batch)
                # print(f" yielded_size: {yielded_size}")
                # print(f"batch shape: {str(batch)}")
                yield batch, batch



    def generate_notes(self):
        generated_notes = self.decoder(np.random.normal(size=(1, self.latent_dim))) \
            .numpy().reshape(self.n_of_unique_notes_classes, self.sequence_length) \
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
        notes = [[] for _ in original_scores]
        durations = [[] for _ in original_scores]
        original_keys = []

        def flatten(t):
            return [item for sublist in t for item in sublist]

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

        selected_songs = []
        print(f"number of scores: {str(original_scores)}")

        for i, song in enumerate(original_scores):
            # song.transpose
            transp_amount = transpose_amount(song)
            key = str(song.analyze('key').transpose(transp_amount))
            print(f"filtering song number: {str(i)} with key {key}")
            if FILTERED_KEYS in key:
                selected_songs.append((song, transp_amount))

        random.shuffle(selected_songs)
        # else:
        # original_keys.append(str(song.analyze('key').transpose(transp_int)))
        for i, (song, transp_int) in enumerate(selected_songs):
            # songs_with
            # original_keys.append(str(song.analyze('key').transpose(transp_int)))
            notes[i] = []
            durations[i] = []

            for element in song:
                if isinstance(element, note.Note):
                    notes[i].append(element.pitch.transpose(transp_int))
                    durations[i].append(element.duration.quarterLength)

                elif isinstance(element, chord.Chord):
                    notes[i].append('.'.join(str(n.transpose(transp_int)) for n in element.pitches))
                    durations[i].append(element.duration.quarterLength)

            # print(str(original_keys[i]))

        # c_notes = [c for (c, k) in zip(notes, original_keys) if (k == 'C major')]
        # c_durations = [c for (c, k) in zip(durations, original_keys) if (k == 'C major')]
        # Map unique notes to integers
        unique_notes = np.unique([i for s in notes for i in s])
        self.note_to_int = dict(zip(unique_notes, list(range(0, len(unique_notes)))))
        self.n_of_unique_notes_classes = len(unique_notes)

        # Map unique durations to integers
        unique_durations = np.unique([i for s in durations for i in s])
        self.duration_to_int = dict(zip(unique_durations, list(range(0, len(unique_durations)))))

        # Print number of unique notes and notes
        print(len(unique_notes))

        # Print number of unique durations
        print(len(unique_durations))

        self.int_to_note = {i: c for c, i in self.note_to_int.items()}
        self.int_to_duration = {i: c for c, i in self.duration_to_int.items()}

        # Define sequence length

        # Define empty arrays for train data
        # train_notes = []
        # train_durations = []
        #
        # test_notes = []
        # test_durations = []

        split_indx = int(TRAIN_SIZE * len(notes))
        train_notes_list = flatten(notes[:split_indx])
        test_notes_list = flatten(notes[split_indx:])

        train_durations_list = flatten(durations[:split_indx])
        test_durations_list = flatten(durations[split_indx:])

        # Construct training sequences for notes and durations

        self.train_notes, self.train_durations = self.create_sequence(train_notes_list, train_durations_list)
        self.test_notes, self.test_durations = self.create_sequence(test_notes_list, test_durations_list)
        # for s in range(len(notes_list)):
        #     note_list = [self.note_to_int[c] for c in notes_list[s]]
        #     duration_list = [self.duration_to_int[d] for d in durations[s]]
        #     for i in range(len(note_list) - self.sequence_length):
        #         train_notes.append(note_list[i:i + self.sequence_length])
        #         train_durations.append(duration_list[i:i + self.sequence_length])

        with open(DATA_TRAIN_NOTES_PATH, 'wb') as filepath:
            pickle.dump(self.train_notes, filepath)

        with open(DATA_TRAIN_DURATIONS_PATH, 'wb') as filepath:
            pickle.dump(self.train_durations, filepath)

        with open(DATA_TEST_NOTES_PATH, 'wb') as filepath:
            pickle.dump(self.test_notes, filepath)

        with open(DATA_TEST_DURATIONS_PATH, 'wb') as filepath:
            pickle.dump(self.test_durations, filepath)

        with open(DATA_INT_TO_NOTE_PATH, 'wb') as filepath:
            pickle.dump(self.int_to_note, filepath)

        with open(DATA_INT_TO_DURATION_PATH, 'wb') as filepath:
            pickle.dump(self.int_to_duration, filepath)

        # self.int_to_note = self.int_to_note
        # self.int_to_duration = self.int_to_duration

    def create_sequence(self, notes_list, durations_list):
        print(str(notes_list))
        print(str(durations_list))
        notes_sequence = []
        durations_sequence = []
        note_list = [self.note_to_int[c] for c in notes_list]
        duration_list = [self.duration_to_int[d] for d in durations_list]

        for i in range(len(note_list) - self.sequence_length):
            notes_sequence.append(note_list[i:i + self.sequence_length])
            durations_sequence.append(duration_list[i:i + self.sequence_length])

        # for s in range(len(notes_list)):
        #     print(f"s: {str(s)}")
        #     print(f"notes_list[s]: {str(notes_list[s])}")
        #     print(f"self.int_to_note: {str(self.int_to_note)}")
        #     # for c in
        #     for c in notes_list:
        #         x = self.note_to_int[c]
        #         print(x)

        # for i in range(len(note_list) - self.sequence_length):
        #     notes_sequence.append(note_list[i:i + self.sequence_length])
        #     durations_sequence.append(duration_list[i:i + self.sequence_length])

        return notes_sequence, durations_sequence

    def prepare_data(self, data):
        # print("trainNotesFlat: " + str(train_notes))

        # train_notes_categorical_not_transposed = tf.keras.utils.to_categorical(self.train_notes, dtype="float16")
        # Convert data to numpy array of type float
        # trainNotes = np.array(trainNotes, np.float32)
        notes_categorical = tf.keras.utils.to_categorical(data, dtype="float16")  # .transpose(0, 2, 1)

        # print(f"train_notes_categorical.device: {train_notes_categorical.device}")
        n_samples = notes_categorical.shape[0]
        self.n_of_unique_notes_classes = notes_categorical.shape[2]
        self.input_dim = self.n_of_unique_notes_classes * self.sequence_length
        # Flatten sequence of notes into single dimension
        return notes_categorical.reshape(n_samples, self.input_dim)
        # print("type(train_notes_flattened): " + str(type(train_notes_flattened)))
        # self.numpy_dataset = train_notes_flattened
        # self.tensor_dataset = tf.data.Dataset.from_tensors(tensors=(train_notes_flattened, train_notes_flattened))

        # return tensor_dataset, input_dim, train_durations, sequence_length, int_to_note, int_to_duration, n_notes

    def train(self, checkpoint_path=None):
        # Define number of samples, notes and notes, and input dimension
        # filepath = CHECKPOINTS + "weights-improvement-{epoch:02d}-{loss:.4f}-{categorical_accuracy:.4f}-bigger.hdf5"
        # filepath = "weights-improvement-epoch:{epoch:02d}-loss:{loss:.4f}-cat_acc:{categorical_accuracy:.4f}.hdf5"

        if checkpoint_path:
            self.model.load_weights(checkpoint_path)
            nb_epoch = int(os.path.basename(checkpoint_path).split("=")[1].split("-")[0])

        else:
            nb_epoch = 0
        # filepath = os.path.join(CHECKPOINT, "epoch={epoch:03d}-loss={loss:.4f}-val_loss={val_loss:.4f}.hdf5")
        filepath = os.path.join(CHECKPOINT, "epoch={epoch:03d}-loss={loss:.4f}.hdf5")

        # filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        log = tf.keras.callbacks.TensorBoard(log_dir=LOG + " " + self.model_type)

        callbacks_list = [checkpoint, log]
        # history = self.model.fit(network_input, network_output, epochs=EPOCHS, batch_size=128, callbacks=callbacks_list)
        # model.save(MODEL_DIR_PATH + MODEL_NAME + "_" + CURR_DT + ".hdf5")

        self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001))
        # self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=[tf.keras.losses.BinaryCrossentropy()])
        # )

        # self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
        # Train autoencoder
        self.model.summary()
        print(MODEL_DIR_PATH + MODEL_NAME + "_" + CURR_DT + ".hdf5")
        # history = self.model.fit(self.trainNotesFlat, self.trainNotesFlat, epochs=1)
        # history = self.model.fit(self.trainNotesFlat, self.trainNotesFlat, epochs=500, callbacks=callbacks_list, batch_size=8)
        # tensor_dataset = tf.data.Dataset.from_tensors((self.trainNotesFlat, self.trainNotesFlat))

        # history = self.model.fit(x=self.data_generator(),
        #                          epochs=EPOCHS,
        #                          callbacks=callbacks_list,
        #                          batch_size=BATCH_SIZE,
        #                          steps_per_epoch=self.steps_per_epoch)
        # self.model.fit_generator(self.data_generator(self.train_dataset),
        #                          epochs=EPOCHS,
        #                          callbacks=callbacks_list,
        #                          initial_epoch=nb_epoch,
        #                          validation_data=self.data_generator(self.test_dataset),
        #                          steps_per_epoch=self.steps_per_epoch)
        # batch_size=BATCH_SIZE,

        self.model.fit(self.train_dataset, self.train_dataset,
                       epochs=EPOCHS,
                       callbacks=callbacks_list,
                       initial_epoch=nb_epoch,
                       validation_data=(self.test_dataset, self.test_dataset),
                       steps_per_epoch=self.steps_per_epoch)
        # print(history.history)
        print(MODEL_DIR_PATH + MODEL_NAME + "_" + CURR_DT + ".hdf5")
        self.model.save(os.path.join(MODEL_DIR_PATH, MODEL_NAME + "_" + CURR_DT + ".hdf5"))
# %%

class ModelFactory:
    def factory(self, model_type, use_computed_values):
        if model_type == AUTOENCODER:
            if use_computed_values:
                model = MusicAutoencoder(model_type, 2, 32, COMPUTED_NOTES_PATH, COMPUTED_DURATIONS_PATH,
                                         COMPUTED_TEST_NOTES_PATH,
                                         COMPUTED_TEST_DURATIONS_PATH, COMPUTED_INT_TO_NOTE_PATH,
                                         COMPUTED_INT_TO_DURATION_PATH)
            else:
                model = MusicAutoencoder(model_type, 2, 32)
            return model


# %%

modelFactory = ModelFactory()
music_autoencoder = modelFactory.factory(MODEL_NAME, USE_COMPUTED_VALUES)
if USE_SAVE_POINT:
    music_autoencoder.train(SAVE_POINT)
else:
    music_autoencoder.train()
music_autoencoder.generate_notes()

# %%
