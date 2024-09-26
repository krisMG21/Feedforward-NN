import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def paint_number(index, X):
    '''
    index: int, X: [int, int] --> null
    Given X dataset and an index, prints on screen the number that it is stored
    '''
    current_image = X[:, index, None]
    current_image = current_image.reshape((28, 28)) * 255

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')

    plt.ion()  # Turn on interactive mode
    plt.show(block=False)  # Show the plot without blocking, and wait for user input
    plt.pause(0.5)  # Pause to allow the plot to update
    time.sleep(1)  # Display for 5 seconds (or adjust as needed)
    plt.close()  # Close the plot


