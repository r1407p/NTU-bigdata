import pandas as pd
import ctypes
import numpy as np
import os

# 動態取得 .so 的路徑（跟 eval.py 同資料夾）
LIB_PATH = os.path.join(os.path.dirname(__file__), 'grader.cpython-310-x86_64-linux-gnu.so')
lib = ctypes.CDLL(LIB_PATH)

# 假設 .so 提供一個 score 函數
lib.score.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.score.restype = ctypes.c_double

def score(labels):
    arr = np.array(labels, dtype=np.float64)
    c_arr = (ctypes.c_double * len(arr))(*arr)
    return lib.score(c_arr, len(arr))

# 讀取 public_submission.csv 並計算分數
submission = pd.read_csv("public_submission.csv").sort_values("id").reset_index(drop=True)
labels_pred = submission["label"].tolist()

print(f"Score: {score(labels_pred):.4f}")
