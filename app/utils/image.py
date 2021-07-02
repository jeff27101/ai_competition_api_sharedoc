import numpy as np
import base64
import cv2


def normalize(y, mean, std):
    x = y.copy()
    for i in [0, 1, 2]:
        x[:, i, :, :] = (y[:, i, :, :] - mean[i]) / std[i]
    return x


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def base64_to_np_array(image_64_encoded):
    """ Convert base64 to numpy.ndarray for cv2.

    @param:s
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    return image


def preprocessing(image, imsize=224):
    height, width, _ = image.shape
    size = max((height, width))
    img = cv2.copyMakeBorder(image, 0, size - height, 0, size - width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img = cv2.resize(img, (imsize, imsize), cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype(np.float32)
    img = np.moveaxis(img, -1, 0)
    img = np.expand_dims(img, axis=0)
    # to 0..1 (ToTensor())
    img = np.divide(img, 255)
    # normalize
    img = normalize(img, [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    return img
