def generate_batches(datas, labels, batch_size=BATCH_SIZE):
    number_of_samples = len(datas)
    number_of_batch = number_of_samples / batch_size
    data_batches = numpy.split(datas, number_of_batch)
    label_batches = numpy.split(labels, number_of_batch)
    batches = [dict(data=data_batches[index], label=label_batches[index]) for index in range(number_of_batch)]
    return batches

def train(datas, labels, updater=ADAM, loss=CROSS_ENTROPY, epochs=EPOCHS, rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2, epsilon=EPSILON, model='model.params', model_path=os.path.dirname(os.path.realpath(__file__))):
    epoch_path = os.path.join(model_path, 'epochs')
    tensors = dict(input=theano.tensor.tensor4(dtype='float32'), output=theano.tensor.matrix(dtype='int8'))
    network = create_network(variable=tensors['input'])
    predictions = lasagne.layers.get_output(layer_or_layers=network)
    losses = loss(predictions=predictions, targets=tensors['output']).mean()
    parameters = lasagne.layers.get_all_params(layer=network, trainable=True)
    updates = updater(loss_or_grads=losses, params=parameters, learning_rate=rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    trainer = theano.function(inputs=[tensors['input'], tensors['output']], outputs=losses, updates=updates)
    batches = generate_batches(datas=datas, labels=labels)
    for epoch in range(epochs):
        print('Epoch {e}:'.format(e=(epoch+1)))
        number_of_batch = len(batches)
        for batch_index in range(number_of_batch):
            batch = batches[batch_index]
            batch_loss = trainer(batch['data'], batch['label'])
            print('Batch {b}: Loss = {l:.5f}'.format(b=(batch_index+1), l=batch_loss))
        epoch_file = 'epoch_{e}.params'.format(e=(epoch+1))
        epoch_model = os.path.join(epoch_path, epoch_file)
        save_network_as_model(network, epoch_model)
    trained_model_file = os.path.join(model_path, model)
    save_network_as_model(network, trained_model_file)
