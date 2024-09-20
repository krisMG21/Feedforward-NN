# Feedforward-NN

> Adaptable ( I/O-wise ) neural network written in Python

## What is this?

This is a simple neural network written in Python.
It is designed to be adaptable to any input and output data.

It's derive from a MNIST neural network made from scratch,
in a code tutorial + concept explanation [video](https://youtu.be/w8yWXqWQYmU?si=vzrMdSi5JYNYPyKD)
from [Samson Zhang](https://www.samsonzhang.com/).

Given a set of clasified data, the network will train on it,
and each iteration it will be able to classify new data more
precisely.

## How does it work?

The network is made up of an input layer, a hidden layer, and
an output layer. The input layer receives the input data, and it
propagates to the forward layers until it reaches the output.

Effectively, the network is stored in a bunch of matrices, one
for each layer, another one for each connection between layers,
all of them storing the weights and biases of each neuron.

This means the input 'flows' through the network, like power
through a circuit, getting amplified or attenuated by the cables,
collected and modified by the neurons, which pass them on, finally
reaching the output.

![gif](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi.makeagif.com%2Fmedia%2F11-07-2017%2FYSd8yg.gif&f=1&nofb=1&ipt=5f1fc71fb28fd4694c2500d1d2acceaa23e1d23376523750c95584876b9625e8&ipo=images)

> [!NOTE]
> **Source**
> From 3Blue1Brown' video [here](https://www.youtube.com/watch?v=aircAruvnKk&t=135s),
> he has a full course on neural networks, amazingly animated.

This transforms the values of the input all across the network,
and the output is the result of the last neuron (0 to 1) or
the most 'powered' neuron's index in the last later (0 to N).

## How to use it?

### Configuration

To configure the network, you need to modify the parameters in
the `config.json` file, to the corresponding number of inputs,
hidden layer's size, and outputs.

For the moment, the depth of the network is not configurable,
given that a single hidden layer is performant enough, and the
increase in depth (apart from the tedious refactoring) would
not be that much distinguishable.

### Training

To train the network, you need to provide it with a set of data
with it's corresponding labels, in the form of a .csv file.

The first column of the .csv file should be the output data or
label, and the following columns should be the input data.

If you have a .csv file with the following data
(for example, parity of the input data):

| Output | Input 1 | Input 2 | Input 3 |
| ------ | ------- | ------- | ------- |
| 0      | 1       | 0       | 1       |
| 1      | 0       | 1       | 0       |
| 0      | 0       | 0       | 0       |
| 1      | 1       | 1       | 1       |
| 0      | 0       | 1       | 1       |
| ...    | ...     | ...     | ...     |

You can train the network with the following command:

```bash
python3 main.py ./train.csv --mode train
```

> [!NOTE]
> The state of the network will be saved into a
> nn_weights.json, if you want to reset the network
> move, rename or delete the file

### Testing

To test the network, you need to provide it with a set of data
in the same format as before.

For example, if you have a .csv file with the following data:

| Output | Input 1 | Input 2 | Input 3 |
| ------ | ------- | ------- | ------- |
| 1      | 1       | 0       | 0       |
| 1      | 0       | 0       | 1       |
| 0      | 1       | 1       | 0       |
| ...    | ...     | ...     | ...     |

You can test the network with the following command:

```bash
python3 main.py ./test.csv --mode test
```

For the moment, the tests consist of:

- A test on the first 20 predictions
- A test on the accuracy on the whole dataset
- A visualization and count of all the fails

> [!NOTE]
> Both the training and testing data should be in the same format
> (number of columns, csv file, etc.), as the configuration is set
> to.

> [!NOTE]
> Names of the csv are not important, as long as the train and test
> data sets are different (the model is trained on one, tested on another).

### TODO

- [x] Revise the code and documentation
- [x] Implement save and load of the network
- [ ] Implement a way to visualize the network
- [ ] Implement a way to export / import the network

## Contributing

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update tests as appropriate.

## Author

- **Cristian Marquez** - *Initial work* - [krisMG21](https://github.com/krisMG21)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.
