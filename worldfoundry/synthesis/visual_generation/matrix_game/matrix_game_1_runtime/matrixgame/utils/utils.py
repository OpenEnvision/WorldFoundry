from PIL import Image


def black_image(width, height):
    """generate a black image

    Args:
        width (int): image width
        height (int): image height

    Returns:
        _type_: a black image
    """
    black_image = Image.new("RGB", (width, height), (0, 0, 0))
    return black_image
