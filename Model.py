def create_network(variable=None, droput=DROPOUT, activation=LEAKY_RECTIFY, classifier=SOFTMAX, weight=HE_NORMAL, bias=CONSTANT, image_shape=IMAGE_SHAPE):
    channel, height, width = image_shape
    input_layer = lasagne.layers.InputLayer(shape=(None, channel, height, width), input_var=variable)
    conv1 = lasagne.layers.Conv2DLayer(incoming=input_layer, num_filters=8, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    conv2 = lasagne.layers.Conv2DLayer(incoming=conv1, num_filters=8, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    pool1 = lasagne.layers.Pool2DLayer(incoming=conv2, pool_size=(2, 2), stride=(2, 2), pad=0)
    drop1 = lasagne.layers.DropoutLayer(incoming=pool1, p=droput)
    conv3 = lasagne.layers.Conv2DLayer(incoming=drop1, num_filters=16, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    conv4 = lasagne.layers.Conv2DLayer(incoming=conv3, num_filters=16, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    pool2 = lasagne.layers.Pool2DLayer(incoming=conv4, pool_size=(2, 2), stride=(2, 2), pad=0)
    drop2 = lasagne.layers.DropoutLayer(incoming=pool2, p=droput)
    fc = lasagne.layers.DenseLayer(incoming=drop2, num_units=len(LABELS), nonlinearity=classifier, W=weight, b=bias)
    return fc

def load_network_from_model(network, model):
    with open(model, 'r') as model_file:
        parameters = pickle.load(model_file)
    lasagne.layers.set_all_param_values(layer=network, values=parameters)

def save_network_as_model(network, model):
    parent_directory = os.path.abspath(model + "/../")
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    parameters = lasagne.layers.get_all_param_values(layer=network)
    with open(model, 'w') as model_file:
        pickle.dump(parameters, model_file)
