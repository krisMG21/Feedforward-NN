import numpy as np
import pandas as pd
import json
import src.lib as nn
import argparse

config = json.load(open("config.json"))

def prepare_data(csv_file):
    # Load and preprocess data
    raw_data = pd.read_csv(csv_file)
    raw_data = np.array(raw_data)
    n, m = raw_data.shape

    # Shuffle data
    np.random.shuffle(raw_data)
    data = raw_data.T

    Y = data[0]
    X = data[1:]
    X = X / X.max()  # Normalize data

    return n, m, X, Y

def test_config(Y):
    # Check if the number of inputs and outputs match the configuration
    in_size = config["input_size"]
    out_min, out_max = config["output_range"]
    x_min, x_max = Y.min().item(), Y.max().item()

    if x_max > out_max or x_min < out_min or in_size != config["input_size"]:
        print("Error: Number of inputs and outputs does not match the configuration")
        print("Expected: ", config["input_size"], config["output_range"])
        print("Got: ", in_size, [x_max, x_min])
        exit()

def main(csv_file, mode):
    n, m, X, Y = prepare_data(csv_file)

    test_config(Y)

    # Initialize weights and biases
    W1, b1, W2, b2 = np.array(""), np.array(""), np.array(""), np.array("")

    # Train or test
    if mode == 'train':
        print("Training...")
        iterations = config["iterations"]
        W1, b1, W2, b2 = nn.gradient_descent(X, Y, 0.10, iterations, m)
    elif mode == 'test':
        W1, b1, W2, b2 = nn.init_params()

        # See 20 first (if possible) predictions
        if config["show_first"] > 0:
            print("="*100 + "\nFirst predictions:\n" + "="*100)
            for i in range(min(config["show_first"], n)):
                nn.test_prediction(i, W1, b1, W2, b2, X, Y)

        # Test fails
        if config["show_fails"]:
            print("="*100 + "\nFails:\n" + "="*100)
            nn.show_fails(W1, b1, W2, b2, X, Y)

        # Test accuracy
        if config["show_accuracy"]:
            print("="*100 + "\nAccuracy:\n" + "="*100)
            _, _, _, A2 = nn.forward_prop(W1, b1, W2, b2, X)
            print(nn.get_accuracy(nn.get_predictions(A2), Y))

    else:
        print("Error: Mode must be 'train' or 'test'")
        exit()

    # Serializing of network status
    data = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist()
    }

    # Serialize weights
    json_data = json.dumps(data, indent=2)
    with open("nn_weights.json", "w") as json_file:
        json_file.write(json_data)

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to the CSV file for training/testing")
    parser.add_argument("--mode", choices=['train', 'test'], required=True, help="Mode: 'train' or 'test'")
    args = parser.parse_args()
    main(args.csv_file, args.mode)
