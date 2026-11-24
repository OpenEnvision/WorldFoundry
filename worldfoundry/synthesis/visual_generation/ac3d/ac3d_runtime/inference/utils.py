from PIL import Image


def stack_images_horizontally(image1: Image.Image, image2: Image.Image) -> Image.Image:
    # Ensure both images have the same height
    height = max(image1.height, image2.height)
    width = image1.width + image2.width

    # Create a new blank image with the combined width and the maximum height
    new_image = Image.new("RGB", (width, height))

    # Paste the images into the new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))

    return new_image
