# MNIST-NN

> Implemented from main branch for MNIST dataset

## Features

The available datasets are linked in the [DATASETS.md](DATASETS.md) file.

If the datasets fall short for you, you can produce your own datasets!
With the [img_interpreter.py](img_interpreter.py) file, you can convert your
images into a .csv file.

### Converting images to .csv

In the Images_for_csv folder, you can find a gimp file prepared for drawing
suitable numbers, which ```img_interpreter.py``` will read and convert into
a .csv file.

With the code as it is, the number the image represents is read from the
first character of the filename, so the 1 images must be named 1_0.png,
1_1.png, 1number.png, and so on.

> [!NOTE]
> The folder name is not important, but needs to be the same as the code
> in ```img_interpreter.py``` tries to read the images from.

The use of the source code is the same as in the main branch, but i'll
keep it here too, just in case.

## Usage

You can train the network with the following command:

```bash
python3 main.py ./train.csv --mode train
```

You can test the network with the following command:

```bash
python3 main.py ./test.csv --mode test
```

## Author

- **Cristian Marquez** - *Initial work* - [krisMG21](https://github.com/krisMG21)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.
