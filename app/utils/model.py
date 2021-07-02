from . import image


def inference(ort_session, img):
    inputs = {ort_session.get_inputs()[0].name: img}
    predicts = ort_session.run(None, inputs)[0][0]
    return image.softmax(predicts)
