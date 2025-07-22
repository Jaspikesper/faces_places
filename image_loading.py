import numpy as np
import os
from PIL import Image

def read_pgm(filename, folder_num=None, photo_num=None):
    if folder_num is not None and photo_num is not None:
        path = os.path.join('face_data', f's{folder_num}', f'{photo_num}.pgm')
    elif folder_num is None and photo_num is None:
        path = filename if filename.endswith('.pgm') else f'{filename}.pgm'
    else:
        raise ValueError("Provide either 1 or 3 positional arguments.")
    with open(path, 'rb') as f:
        assert f.readline()[:2] == b'P5'
        while True:
            line = f.readline()
            if not line.startswith(b'#'):
                break
        width, height = map(int, line.strip().split())
        maxval = int(f.readline())
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width))
    return img

def crop_and_resize(arr, out_size=256):
    min_side = min(arr.shape)
    y, x = arr.shape
    y0 = (y - min_side) // 2
    x0 = (x - min_side) // 2
    cropped = arr[y0:y0+min_side, x0:x0+min_side]
    pil_img = Image.fromarray(cropped)
    resized = pil_img.resize((out_size, out_size), Image.LANCZOS)
    return np.array(resized, dtype=np.uint8)

def save_pgm(filename, arr):
    h, w = arr.shape
    with open(filename, 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (w, h))
        f.write(arr.tobytes())

def read_all_faces(face_data_dir='face_data', crop_top=10, crop_bottom=10):
    data = []
    for subject in os.listdir(face_data_dir):
        for photo_num in [1, 2, 3]:
            img = read_pgm(str(photo_num), int(subject[1:]) if subject.startswith('s') else subject, photo_num)
            img = img[crop_top:img.shape[0]-crop_bottom, :]
            data.append(img.flatten())
    A = np.array(data)
    return A