import numpy as np
import pandas as pd
import json
import lib as nn
import argparse

def main(csv_file, mode):
    # Load configuration
    config = json.load(open("config.json"))

    # Load and preprocess data
    data = pd.read_csv(csv_file)
    data = np.array(data)
    m, n = data.shape

    # Shuffle data
    np.random.shuffle(data)
    data_train = data[0:m].T

    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / X_train.max()  # Normalize

    # Check if the number of inputs and outputs match the configuration
    outputs = Y_train.max() - Y_train.min() + 1
    inputs = X_train.shape[0]

    if outputs != config["output_size"] or inputs != config["input_size"]:
        print("Error: Number of inputs and outputs does not match the configuration")
        print("Expected: ", config["input_size"], config["output_size"])
        print("Got: ", inputs, outputs)
        exit()


    # Initialize weights and biases
    W1, b1, W2, b2 = np.array(""), np.array(""), np.array(""), np.array("")

    if mode == 'train':
        print("Training...")
        iterations = int(input("Number of iterations desired: "))
        W1, b1, W2, b2 = nn.gradient_descent(X_train, Y_train, 0.10, iterations, m, config)
    elif mode == 'test':
        W1, b1, W2, b2 = nn.init_params(config)

        # See 20 first (if possible) predictions
        print("="*100 + "\nFirst 20 predictions:\n" + "="*100)
        for i in range(min(20, m)):
            nn.test_prediction(i, W1, b1, W2, b2, X_train, Y_train)

        # Test accuracy
        print("="*100 + "\nAccuracy:\n" + "="*100)
        _, _, _, A2 = nn.forward_prop(W1, b1, W2, b2, X_train)
        print(nn.get_accuracy(nn.get_predictions(A2), Y_train))

        # See fails
        print("="*100 + "\nFails:\n" + "="*100)
        nn.show_fails(W1, b1, W2, b2, X_train, Y_train)

    # Serializing of network status
    data = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist()
    }

    json_data = json.dumps(data, indent=2)
    with open("nn_weights.json", "w") as json_file:
        json_file.write(json_data)

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to the CSV file for training/testing")
    parser.add_argument("--mode", choices=['train', 'test'], required=True, help="Mode: 'train' or 'test'")
    args = parser.parse_args()
    main(args.csv_file, args.mode)
