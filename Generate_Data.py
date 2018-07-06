import Image_Generator as imgen
import pylab as plt
import numpy as np
import sys
import os
import argparse


# Generate specified amount/type of images
def generate(_):
    # Create a data directory if not already there
    if not os.path.exists("./data"):
        os.makedirs("./data")

    # Instantiate generator
    generator = imgen.Image_Generator()

    # Create specified amount/type of images with random onehot pixels
    for x in range(0, FLAGS.num_images):
        image = generator.create_image(FLAGS.num_onehot, FLAGS.mean,
                                       FLAGS.std_deviation,
                                       (FLAGS.height, FLAGS.width),
                                       FLAGS.noise_func)

        # Save images to the ./data directory
        plt.imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=1,
                   pad_inches=0, transparent=True, dpi=136.9)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("data/img" + str(x) + ".png", bbox_inches="tight")


# Generate 1000 images of each noise function for GAN
def generate_dataset_for_GAN():
    # Create data directories if not already there
    if not os.path.exists("./gaussian_images"):
        os.makedirs("./gaussian_images")
    if not os.path.exists("./sinusoidal_images"):
        os.makedirs("./sinusoidal_images")

    # Instantiate generator
    generator = imgen.Image_Generator()

    plt.rcParams['savefig.pad_inches'] = 0
    plt.autoscale(tight=True)

    # Create 1000 Gaussian images with random onehot pixels
    for x in range(0, 1000):
        image = generator.create_image(np.random.randint(6), 0.25,
                                       0.0625, (64, 64), "gaussian")

        # Save images to the ./gaussian_images directory
        plt.imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("gaussian_images/gaussian" + str(x) + ".png",
                    bbox_inches="tight", pad_inches=0,
                    transparent=True, dpi=136.9)

    # Create 1000 sinusoidal images with random onehot pixels
    for x in range(0, 1000):
        image = generator.create_image(np.random.randint(6), 0.25,
                                       0.0625, (512, 512), "sinusoidal")

        # Save images to the ./sinusoidal_images directory
        plt.imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("sinusoidal_images/sinusoidal" + str(x) + ".png",
                    bbox_inches="tight", pad_inches=0,
                    transparent=True, dpi=136.9)


# Main function to accept user input and generate images
if __name__ == "__main__":
    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Establish default arguments
    parser.add_argument("--num_onehot", type=int, default=16,
                        help="Number of onehot pixels to generate")

    parser.add_argument("--num_images", type=int, default=32,
                        help="Number of images to generate")

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

    # Generate images
    generate([sys.argv[0]] + unparsed)
    # generate_dataset_for_GAN()
