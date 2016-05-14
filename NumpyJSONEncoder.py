import json
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """Extended JSON encoder default method to handle numpy classes. This is incomplete - add more as needed."""
    def default(self, obj):

        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()

        if isinstance(obj, np.int32):
            return int(obj)

        if isinstance(obj, np.float64):
            return float(obj)

        return json.JSONEncoder.default(self, obj)