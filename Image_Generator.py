import pylab as plt
import numpy as np
import sys
import argparse


# Synthetic image generator
class Image_Generator:
    # Class has no member variables
    def __init__(self):
        pass

    # Create onehot vector
    def __make_onehot(self, num_onehot, height, width):
        onehot = []

        # Populate onehot vector with random coordinates
        while len(onehot) < num_onehot:
            candidate_onehot = [np.random.randint(height),
                                np.random.randint(width)]

            if candidate_onehot not in onehot:
                onehot.append(candidate_onehot)

        return onehot

    # Create an image with Gaussian-distribution noise background
    def __create_gaussian_noise_bg(self, mean, std_deviation, size_tuple):
        # Return Gaussian-distribution matrix
        return np.random.normal(mean, std_deviation, size_tuple)

    # Create array of sin vals according to
    # amplitude (min/max vals) and period (num of vals)
    def __create_sin_vals(self, amplitude, period):
        return np.sin(np.array(np.linspace(amplitude,
                                           180 - amplitude,
                                           num=period)) * np.pi / 180.)

    # Create an image with sinusoidal-distribution noise background
    def __create_sinusoidal_noise_bg(self, mean, std_deviation, size_tuple):
        # Start with a Gaussian noise bg
        image = self.__create_gaussian_noise_bg(mean,
                                                std_deviation,
                                                size_tuple)

        # Create sin properties
        # The generated numbers are proportional, not equal, to the properties
        amplitude = round(np.random.uniform(15, 60), 2)
        period = np.random.randint(10, 31)
        phase_shift = np.random.randint(period + 1)

        # Create array of sin vals to act as bias for Gaussian bg
        sin_vals = self.__create_sin_vals(amplitude, period)

        # Multiply Gaussian noise by sin vals, adjusting for period/phase shift
        for row in range(size_tuple[0]):
            for col in range(size_tuple[1]):
                index = (int(col / 3) + phase_shift) % period
                image[row][col] = image[row][col] * sin_vals[index]

        # Return sinusoidal-distribution matrix
        return image

    # Create an image with specified noise and populate with random onehots
    def create_image(self, num_onehot, mean,
                     std_deviation, size_tuple, noise_func):
        # Create specified noise background
        # If the function is not approved, raise NotImplementedError
        if noise_func == "gaussian":
            image = self.__create_gaussian_noise_bg(mean,
                                                    std_deviation,
                                                    size_tuple)
        elif noise_func == "sinusoidal":
            image = self.__create_sinusoidal_noise_bg(mean,
                                                      std_deviation,
                                                      size_tuple)
        else:
            print("""Noise function not implemented.
                  Please select "gaussian" or "sinusoidal".""")
            raise NotImplementedError

        if num_onehot > 0:
            # Create onehot vector
            onehot = self.__make_onehot(num_onehot,
                                        size_tuple[0], size_tuple[1])

            # Set all the onehot coords to 1
            for coord in onehot:
                image[coord[0], coord[1]] = 1

        # Return image matrix
        return image


# Main function for testing
def example_usage(_):
    # Initialize a generator
    generator = Image_Generator()

    # Create specified image matrix
    image = generator.create_image(FLAGS.num_onehot,
                                   FLAGS.mean,
                                   FLAGS.std_deviation,
                                   (FLAGS.height, FLAGS.width),
                                   FLAGS.noise_func)

    # Display the image
    plt.imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=1,
               pad_inches=0, transparent=True, dpi=136.9)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Establish default arguments
    parser.add_argument("--num_onehot", type=int, default=16,
                        help="Number of onehot pixels to generate")

    parser.add_argument("--mean", type=float, default=0.25,
                        help="Mean for normal distribution")

    parser.add_argument("--std_deviation", type=float, default=0.0625,
                        help="Standard deviation for normal distribution")

    parser.add_argument("--height", type=int, default=512, help="Image height")

    parser.add_argument("--width", type=int, default=512, help="Image width")

    parser.add_argument("--noise_func", type=str, default="gaussian",
                        help="Noise function: 'gaussian' or 'sinusoidal'")

    # Parse known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Run the example usage function
    example_usage([sys.argv[0]] + unparsed)
