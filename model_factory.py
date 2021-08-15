from autoencoder import MusicAutoencoder
from prepare_data import create_train_data


class ModelFactory:
    def Factory(self, type):
        match type:
            case "autoencoder":
                trainChords, trainDurations, sequenceLength, intToChord, intToDuration = create_train_data()
                return MusicAutoencoder(2, trainChords, sequenceLength, intToChord, intToDuration)
