import readTrafficSigns
import numpy
# from keras.preprocessing.image import ImageDataGenerator
#
# datagen = ImageDataGenerator(
#     rotation_range=30
#     # width_shift_range=0.15,
#     # height_shift_range=0.15,
#     # zoom_range=0.1
# )


class DataSet(object):
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._num_examples = self._images.shape[0]
        assert self._images.shape[0] == self._labels.shape[0]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def shuffle(self):
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]


def read_data_sets(mode):
    class DataSets(object):
        pass

    data_sets = DataSets()

    train_path = './Final_Training/Images'
    # test_path = './Online-Test-sort'
    test_path = './Final_Test/Images'

    if mode == 'train':
        train_images, train_labels = readTrafficSigns.readTrafficSigns(train_path, 'train')
        data_set = DataSet(train_images, train_labels)
        data_set.shuffle()
        x = data_set.images
        y = data_set.labels

        data_sets.validation = DataSet(x[:4000], y[:4000])
        data_sets.train = DataSet(x[4000:], y[4000:])
        # data_sets.train.shuffle()
    else:
        test_images, test_labels = readTrafficSigns.readTrafficSigns(test_path, 'test')
        data_sets.test = DataSet(test_images, test_labels)
        # data_sets.test.shuffle()
    return data_sets
