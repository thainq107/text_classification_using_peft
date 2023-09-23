import json
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

label2id = {
    'negative': '0',
    'positive': '1'
}

def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2

def postprocess_text(predictions, labels):
    predictions = [prediction.strip() for prediction in predictions]
    labels = [label2id[label.strip()] for label in labels]

    for idx in range(len(predictions)):
        if predictions[idx] in label2id:
           predictions[idx] = label2id[predictions[idx]]
        else:
            predictions[idx] = '-100'
    return predictions, labels