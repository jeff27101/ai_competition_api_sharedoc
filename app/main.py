import time

from fastapi import FastAPI
import datetime
import cv2
import json
import onnxruntime
from loguru import logger
import uuid
import utils


CAPTAIN_EMAIL = 'jeff27101@gmail.com'
SALT = 'PinkShark'

# load models
ort_sessions = [
    utils.RunTimeSession(name="MobileNetV3-Large-07",
                         session=onnxruntime.InferenceSession("./models/mobilenetv3_large_07_pretrain.onnx"),
                         img_size=224),
    utils.RunTimeSession(name="MobileNetV3-Large-1",
                         session=onnxruntime.InferenceSession("./models/mobilenetv3_large_10_pretrain.onnx"),
                         img_size=224),
    utils.RunTimeSession(name="EfficientNet-Lite-2",
                         session=onnxruntime.InferenceSession("./models/efficientnet_lite2_pretrain.onnx"),
                         img_size=260)
    ]

with open('./idx_to_class.json') as f:
    idx_to_word = json.load(f)
app = FastAPI()


def predict(image):
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """
    # preprocessing images with different image size
    target_size_set = set([ss.img_size for ss in ort_sessions])
    images = {size: utils.image.preprocessing(image, size) for size in target_size_set}
    predicts = [
        utils.RunTimePredict(name=ss.name,
                             predict=utils.model.inference(ss.session,
                                                           images.get(ss.img_size)),
                             idx_to_word=idx_to_word)
        for ss in ort_sessions
    ]

    # ensemble predictions
    ensemble_predict = utils.RunTimePredict(name="Ensemble",
                                            predict=sum([p.predict for p in predicts])/len(predicts),
                                            idx_to_word=idx_to_word)
    predict_word = ensemble_predict.fetch_predict_word()

    # logging results
    for p in predicts:
        p.log_prediction()
    ensemble_predict.log_prediction()

    if utils.server.check_datatype_to_string(predict_word):
        return predict_word


@app.post('/inference')
def inference(inference: utils.Inference):
    """ API that return your model predictions when E.SUN calls this API. """

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = inference.esun_timestamp
    logger.info(f"receive from esun: {esun_timestamp}")

    image = utils.image.base64_to_np_array(inference.image)
    img_to_save = image.copy()

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = utils.server.generate_server_uuid(CAPTAIN_EMAIL + ts, salt=SALT)
    if inference.retry and inference.retry >= 2:
        logger.warning(
            f"{inference.esun_timestamp} - {inference.esun_uuid} retry {inference.retry} times. please check inference")

    try:
        answer = predict(image)
    except TypeError as type_error:
        logger.error(type_error)
        raise type_error
    except Exception as e:
        logger.error(e)
        raise e
    try:
        cv2.imwrite(f"/data/{int(time.time())}_{uuid.uuid4().hex}_{answer}.jpg", img_to_save)
    except Exception as e:
        logger.error(f"write image error: {e}")

    return {'esun_uuid': inference.esun_uuid,
            'server_uuid': server_uuid,
            'answer': answer,
            'server_timestamp': int(time.time())}


@app.get("/")
def read_root():
    return {"Hello": "World"}
