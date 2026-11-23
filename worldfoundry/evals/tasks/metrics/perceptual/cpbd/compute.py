from math import atan2, pi
from sys import argv
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .octave import sobel
else:
    from octave import sobel
from skimage.feature import canny
from skimage.io import imread

# threshold to characterize blocks as edge/non-edge blocks
THRESHOLD = 0.002

# fitting parameter
BETA = 3.6

# block size
BLOCK_HEIGHT, BLOCK_WIDTH = (64, 64)

# just noticeable widths based on the perceptual experiments
WIDTH_JNB = np.concatenate([5 * np.ones(51), 3 * np.ones(205)])


def compute(image):
    # type: (numpy.ndarray) -> float
    """Compute the sharpness metric for the given data."""

    # convert the image to double for further processing
    image = image.astype(np.float64)

    # edge detection using canny and sobel canny edge detection is done to
    # classify the blocks as edge or non-edge blocks and sobel edge
    # detection is done for the purpose of edge width measurement.
    canny_edges = canny(image)
    sobel_edges = sobel(image)

    # edge width calculation
    marziliano_widths = marziliano_method(sobel_edges, image)

    # sharpness metric calculation
    return _calculate_sharpness_metric(image, canny_edges, marziliano_widths)


def marziliano_method(edges, image):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    Calculate the widths of the given edges.

    :return: A matrix with the same dimensions as the given image with 0's at
        non-edge locations and edge-widths at the edge locations.
    """

    # `edge_widths` consists of zero and non-zero values. A zero value
    # indicates that there is no edge at that position and a non-zero value
    # indicates that there is an edge at that position and the value itself
    # gives the edge width.
    edge_widths = np.zeros(image.shape)

    # find the gradient for the image
    gradient_y, gradient_x = np.gradient(image)

    # dimensions of the image
    img_height, img_width = image.shape

    # holds the angle information of the edges
    edge_angles = np.zeros(image.shape)

    # calculate the angle of the edges
    for row in range(img_height):
        for col in range(img_width):
            if gradient_x[row, col] != 0:
                edge_angles[row, col] = atan2(gradient_y[row, col], gradient_x[row, col]) * (180 / pi)
            elif gradient_x[row, col] == 0 and gradient_y[row, col] == 0:
                edge_angles[row, col] = 0
            elif gradient_x[row, col] == 0 and gradient_y[row, col] == pi / 2:
                edge_angles[row, col] = 90

    if np.any(edge_angles):
        # quantize the angle
        quantized_angles = 45 * np.round(edge_angles / 45)

        for row in range(1, img_height - 1):
            for col in range(1, img_width - 1):
                if edges[row, col] == 1:
                    # gradient angle = 180 or -180
                    if quantized_angles[row, col] == 180 or quantized_angles[row, col] == -180:
                        for margin in range(100 + 1):
                            inner_border = (col - 1) - margin
                            outer_border = (col - 2) - margin

                            # outside image or intensity increasing from left to right
                            if outer_border < 0 or (image[row, outer_border] - image[row, inner_border]) <= 0:
                                break

                        width_left = margin + 1

                        for margin in range(100 + 1):
                            inner_border = (col + 1) + margin
                            outer_border = (col + 2) + margin

                            # outside image or intensity increasing from left to right
                            if outer_border >= img_width or (image[row, outer_border] - image[row, inner_border]) >= 0:
                                break

                        width_right = margin + 1

                        edge_widths[row, col] = width_left + width_right

                    # gradient angle = 0
                    if quantized_angles[row, col] == 0:
                        for margin in range(100 + 1):
                            inner_border = (col - 1) - margin
                            outer_border = (col - 2) - margin

                            # outside image or intensity decreasing from left to right
                            if outer_border < 0 or (image[row, outer_border] - image[row, inner_border]) >= 0:
                                break

                        width_left = margin + 1

                        for margin in range(100 + 1):
                            inner_border = (col + 1) + margin
                            outer_border = (col + 2) + margin

                            # outside image or intensity decreasing from left to right
                            if outer_border >= img_width or (image[row, outer_border] - image[row, inner_border]) <= 0:
                                break

                        width_right = margin + 1

                        edge_widths[row, col] = width_right + width_left

    return edge_widths


def _calculate_sharpness_metric(image, edges, edge_widths):
    # type: (numpy.array, numpy.array, numpy.array) -> numpy.float64

    # get the size of image
    img_height, img_width = image.shape

    total_num_edges = 0
    hist_pblur = np.zeros(101)

    # maximum block indices
    num_blocks_vertically = int(img_height / BLOCK_HEIGHT)
    num_blocks_horizontally = int(img_width / BLOCK_WIDTH)

    #  loop over the blocks
    for i in range(num_blocks_vertically):
        for j in range(num_blocks_horizontally):
            # get the row and col indices for the block pixel positions
            rows = slice(BLOCK_HEIGHT * i, BLOCK_HEIGHT * (i + 1))
            cols = slice(BLOCK_WIDTH * j, BLOCK_WIDTH * (j + 1))

            if is_edge_block(edges[rows, cols], THRESHOLD):
                block_widths = edge_widths[rows, cols]
                # rotate block to simulate column-major boolean indexing
                block_widths = np.rot90(np.flipud(block_widths), 3)
                block_widths = block_widths[block_widths != 0]

                block_contrast = get_block_contrast(image[rows, cols])
                block_jnb = WIDTH_JNB[block_contrast]

                # calculate the probability of blur detection at the edges
                # detected in the block
                prob_blur_detection = 1 - np.exp(-(abs(block_widths / block_jnb) ** BETA))

                # update the statistics using the block information
                for probability in prob_blur_detection:
                    bucket = round(probability * 100)
                    hist_pblur[bucket] += 1
                    total_num_edges += 1

    # normalize the pdf
    if total_num_edges > 0:
        hist_pblur = hist_pblur / total_num_edges

    # calculate the sharpness metric
    return np.sum(hist_pblur[:64])


def is_edge_block(block, threshold):
    # type: (numpy.ndarray, float) -> bool
    """Decide whether the given block is an edge block."""
    return np.count_nonzero(block) > (block.size * threshold)


def get_block_contrast(block):
    # type: (numpy.ndarray) -> int
    return int(np.max(block) - np.min(block))


if __name__ == "__main__":
    input_image = imread(argv[1], mode="L")
    sharpness = compute(input_image)
    print(f"CPBD sharpness for {argv[1]}: {sharpness:f}")
