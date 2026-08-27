import cv2
import numpy as np
import os

def gen_random_data(path="./test", length=200):
    os.makedirs(path, exist_ok=True)
    data_path = os.path.join(path, "data")

    os.makedirs(os.path.join(data_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "radars"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "labels"), exist_ok=True)

    for i in range(length):
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        radar = np.random.rand(360).astype(np.float32)
        label = np.random.rand(2).astype(np.float32)
        cv2.imwrite(os.path.join(data_path, "images", f"{i:03d}.png"), img)
        np.save(os.path.join(data_path, "radars", f"{i:03d}.txt"), radar)
        np.save(os.path.join(data_path, "labels", f"{i:03d}.txt"), label)
