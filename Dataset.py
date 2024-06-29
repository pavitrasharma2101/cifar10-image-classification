
def preprocess(data):
    return data / numpy.float32(256)

def load_batch(batch_file):
    with open(batch_file, mode='rb') as opened_file:
        batch = pickle.load(opened_file)
        labels = batch[b'labels']
        datas = batch[b'data']
        names = batch[b'filenames']
    return names, datas, labels

def load_train_samples(dataset=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cifar10'), labels=LABELS, image_shape=IMAGE_SHAPE):
    number_of_labels = len(labels)
    train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    train_batch_files = [os.path.join(dataset, train_batch) for train_batch in train_batches]
    x_train = []; y_train = []
    for train_batch_file in train_batch_files:
        _, datas, labels = load_batch(train_batch_file)
        number_of_batch_samples = len(datas)
        for index in range(number_of_batch_samples):
            data = preprocess(data=numpy.reshape(datas[index], image_shape))
            label = [1 if labels[index] == j else 0 for j in range(number_of_labels)]
            x_train.append(data); y_train.append(label)
    datas = numpy.array(x_train, dtype=numpy.float32)
    labels = numpy.array(y_train, dtype=numpy.int8)
    return datas, labels

def load_test_samples(dataset_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cifar10'), labels=LABELS, image_shape=IMAGE_SHAPE):
    number_of_labels = len(labels)
    test_batch = 'test_batch'
    test_batch_file = os.path.join(dataset_path, test_batch)
    x_test = []; y_test = []
    _, datas, labels = load_batch(test_batch_file)
    number_of_samples = len(datas)
    for index in range(number_of_samples):
        data = preprocess(data=numpy.reshape(datas[index], image_shape))
        label = [1 if labels[index] == j else 0 for j in range(number_of_labels)]
        x_test.append(data); y_test.append(label)
    datas = numpy.array(x_test, dtype=numpy.float32)
    labels = numpy.array(y_test, dtype=numpy.int8)
    return datas, labels
