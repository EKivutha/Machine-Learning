eKibz, [15.05.20 20:21]
# Backpropagate error and store in neurons

def backward_propagate_error(network, expected):

 for i in reversed(range(len(network))):

  layer = network[i]

  errors = list()

  if i != len(network)-1:

   for j in range(len(layer)):

    error = 0.0

    for neuron in network[i + 1]:

     error += (neuron['weights'][j] * neuron['delta'])

    errors.append(error)

  else:

   for j in range(len(layer)):

    neuron = layer[j]

    errors.append(expected[j] - neuron['output'])

  for j in range(len(layer)):

   neuron = layer[j]

   neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

 

# Update network weights with error

def update_weights(network, row, l_rate):

 for i in range(len(network)):

  inputs = row[:-1]

  if i != 0:

   inputs = [neuron['output'] for neuron in network[i - 1]]

  for neuron in network[i]:

   for j in range(len(inputs)):

    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]

   neuron['weights'][-1] += l_rate * neuron['delta']

 

# Train a network for a fixed number of epochs

def train_network(network, train, l_rate, n_epoch, n_outputs):

 for epoch in range(n_epoch):

  for row in train:

   outputs = forward_propagate(network, row)

   expected = [0 for i in range(n_outputs)]

   expected[row[-1]] = 1

   backward_propagate_error(network, expected)

   update_weights(network, row, l_rate)

 

# Initialize a network

def initialize_network(n_inputs, n_hidden, n_outputs):

 network = list()

 hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]

 network.append(hidden_layer)

 output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]

 network.append(output_layer)

 return network

 

# Make a prediction with a network

def predict(network, row):

 outputs = forward_propagate(network, row)

 return outputs.index(max(outputs))

 

# Backpropagation Algorithm With Stochastic Gradient Descent

def back_propagation(train, test, l_rate, n_epoch, n_hidden):

 n_inputs = len(train[0]) - 1

 n_outputs = len(set([row[-1] for row in train]))

 network = initialize_network(n_inputs, n_hidden, n_outputs)

 train_network(network, train, l_rate, n_epoch, n_outputs)

 predictions = list()

 for row in test:

  prediction = predict(network, row)

  predictions.append(prediction)

 return(predictions)

 

# Test Backprop on Seeds dataset

seed(1)

# load and prepare data

filename = 'seeds_dataset.csv'

dataset = load_csv(filename)

for i in range(len(dataset[0])-1):

 str_column_to_float(dataset, i)

# convert class column to integers

str_column_to_int(dataset, len(dataset[0])-1)

# normalize input variables

minmax = dataset_minmax(dataset)

normalize_dataset(dataset, minmax)

# evaluate algorithm

n_folds = 5

l_rate = 0.3

n_epoch = 500

n_hidden = 5

scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)

print('Scores: %s' % scores)

print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))