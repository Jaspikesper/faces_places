import numpy as np
import os
from PIL import Image


def read_pgm(filename, folder_num=None, photo_num=None, crop_top=10, crop_bottom=10):
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
    # Crop unless image is already square
    if img.shape[0] != img.shape[1]:
        img = img[crop_top:img.shape[0]-crop_bottom, :]
        # Validate after crop
        if img.shape[0] != img.shape[1]:
            raise ValueError(f"Image is not square after cropping: shape={img.shape}")
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

def read_all_faces(face_data_dir='face_data'):
    data = []
    for subject in os.listdir(face_data_dir):
        for photo_num in [1, 2, 3]:
            img = read_pgm(str(photo_num), int(subject[1:]) if subject.startswith('s') else subject, photo_num)
            # Output validation
            if img.ndim != 2:
                raise ValueError(f"Image is not 2D: shape={img.shape}, subject={subject}, photo={photo_num}")
            if img.shape[0] != img.shape[1]:
                raise ValueError(f"Image is not square after loading: shape={img.shape}, subject={subject}, photo={photo_num}")
            if img.shape[0] <= 0 or img.shape[1] <= 0:
                raise ValueError(f"Image has invalid dimensions: shape={img.shape}, subject={subject}, photo={photo_num}")
            data.append(img.flatten())
    A = np.array(data)
    # Validate stacked array
    if A.ndim != 2:
        raise ValueError(f"Output array is not 2D: shape={A.shape}")
    if A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError(f"Output array has invalid shape: {A.shape}")
    return A