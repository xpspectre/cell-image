from PIL import Image, TiffImagePlugin

TiffImagePlugin.WRITE_LIBTIFF = True


def save_tiff(output, img):
    """Save numpy array img as compressed TIFF as output file

    Args:
        output:
        img:

    Returns:

    """
    pil_img = Image.fromarray(img)
    pil_img.save(output, compression='packbits')