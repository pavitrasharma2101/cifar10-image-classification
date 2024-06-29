def predict(data_or_datas, model, image_shape=IMAGE_SHAPE):
    input_tensor = theano.tensor.tensor4(dtype='float32')
    network = create_network(variable=input_tensor)
    load_network_from_model(network=network, model=model)
    prediction = lasagne.layers.get_output(layer_or_layers=network, deterministic=True)
    result = theano.tensor.argmax(prediction, axis=1)
    predictor = theano.function(inputs=[input_tensor], outputs=result)
    if data_or_datas.shape != image_shape:
        datas = data_or_datas
        predictions = predictor(datas)
        return predictions
    else:
        channel, height, width = image_shape
        data = numpy.reshape(data_or_datas, newshape=(1, channel, height, width))
        prediction = predictor(data)
        return prediction

def test(datas, labels, model=os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.params")):
    number_of_samples = len(datas)
    predictions = predict(data_or_datas=datas, model=model)
    accurancy = 0
    for index in range(number_of_samples):
        prediction = predictions[index]
        target = numpy.argmax(labels[index])
        if target == prediction:
            accurancy += 1
    accurancy = (numpy.float32(accurancy) / number_of_samples) * 100
    print('Accurancy: {a:.3f}'.format(a=accurancy))
