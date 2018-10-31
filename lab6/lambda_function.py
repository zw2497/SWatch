import json
from sklearn import preprocessing
from sklearn.externals import joblib
import numpy as np

def sampling(l, n):
    result = []
    for i in range(n):
        result.append(l[int((i / n) * len(l))])
    return result


def lambda_handler(event, context):
    x = event['content']['data']['x']
    y = event['content']['data']['y']
    x = sampling(x, 30)
    y = sampling(y, 30)
    features = x + y
    features = preprocessing.normalize(np.array(features).reshape(1,-1), norm='l2')

    clf = joblib.load('gesture.joblib')
    res = clf.predict(features)
    return {
        "statusCode": 200,
        "body": int(res[0])
    }

# event = {
#   "label": "c",
#   "n": 12,
#   "number": 8,
#   "content": {
#     "data": {
#       "x": [
#   264,
#   256,
#   624,
#   592,
#   528,
#   264,
#   288,
#   576,
#   376,
#   376,
#   120,
#   776,
#   800,
#   608,
#   528,
#   784,
#   304,
#   776,
#   272,
#   384,
#   56,
#   304,
#   88,
#   624,
#   888,
#   1096,
#   840,
#   600,
#   616,
#   344,
#   288,
#   264,
#   344,
#   8,
#   24,
#   296,
#   368,
#   304,
#   112,
#   328,
#   376,
#   616,
#   296,
#   56,
#   8,
#   88,
#   56,
#   288,
#   304,
#   576,
#   40,
#   24,
#   24,
#   16,
#   16,
#   16,
#   32
# ],
#       "y": [
#   264,
#   256,
#   624,
#   592,
#   528,
#   264,
#   288,
#   576,
#   376,
#   376,
#   120,
#   776,
#   800,
#   608,
#   528,
#   784,
#   304,
#   776,
#   272,
#   384,
#   56,
#   304,
#   88,
#   624,
#   888,
#   1096,
#   840,
#   600,
#   616,
#   344,
#   288,
#   264,
#   344,
#   8,
#   24,
#   296,
#   368,
#   304,
#   112,
#   328,
#   376,
#   616,
#   296,
#   56,
#   8,
#   88,
#   56,
#   288,
#   304,
#   576,
#   40,
#   24,
#   24,
#   16,
#   16,
#   16,
#   32
# ]
#     }
#   }
# }
# context = {}
# print(lambda_handler(event,context))
