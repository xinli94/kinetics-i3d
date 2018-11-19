import cv2
import os
import glob
import numpy as np
import subprocess

IN_PATH = '/data5/xin/i3d/HDHIT310Q_4171741/png'
OUT_PATH = '/data5/xin/i3d/HDHIT310Q_4171741/npy'

# IN_PATH = '/data5/xin/i3d/'
# OUT_PATH = '/data5/xin/i3d/npy'

FRAME_RATE = 25
IMAGE_SIZE = 224
FRAME_PER_CHUNK = 100
OVERLAP = 20
EXPECTED_SHAPE = (FRAME_PER_CHUNK, IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZE = 8

def resize(im, target_size=IMAGE_SIZE):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_scale = float(IMAGE_SIZE) / float(im_size_min)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im

def fix(chunk):
    if chunk.shape != EXPECTED_SHAPE:
        tmp = np.zeros(EXPECTED_SHAPE)
#         tmp[:chunk.shape[0], :chunk.shape[1], :chunk.shape[2], :chunk.shape[3], :chunk.shape[4]] = chunk
        tmp[:chunk.shape[0], :chunk.shape[1], :chunk.shape[2], :chunk.shape[3]] = chunk
        chunk = tmp
    assert(chunk.shape == EXPECTED_SHAPE)
    print('>>>>>>>>>>>', chunk.shape)
    return chunk

image_paths = glob.glob(os.path.join(IN_PATH, '*.png'))    
image_paths.sort(key=lambda x: int(os.path.splitext(x.split('-')[-1])[0]) )
# print(">>>>>>>>>>> image_paths", image_paths)
# print('>>>>>>>>>', len(image_paths))

result = []
chunk = []
chunk_idx = 0
start_idx = 0
for image_idx, image_path in enumerate(image_paths):
    im = cv2.imread(image_path)
    im = resize(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    h, w, c = im.shape
    half = IMAGE_SIZE / 2
    h_start = max(0, h / 2 - half)
    w_start = max(0, w / 2 - half)
    
    im_array = np.array(im)[h_start:h_start+IMAGE_SIZE, w_start:w_start+IMAGE_SIZE, :]
    im_array = (np.float32(im_array) / 255.0 - 0.5) * 2
    assert(im_array.shape == (IMAGE_SIZE, IMAGE_SIZE, 3))
    
    chunk.append(im_array)
    
    if len(chunk) == FRAME_PER_CHUNK:
#         result.append(np.expand_dims(np.array(chunk), axis=0))
        result.append(fix(np.array(chunk)))
        chunk = chunk[-20:]
    
    if len(result) == BATCH_SIZE:
        end_idx = image_idx+1
        save_path = os.path.join(OUT_PATH, 'chunk_{}_start_{}_end_{}.npy').format(chunk_idx, start_idx, end_idx)
        np.save(save_path, np.stack(result, axis=0))
        # print('>>>>>>>>>> saved to {}'.format(save_path))
        # print('>>>>>>>>>>> here', np.stack(result, axis=0).shape)
        chunk_idx += 1
        result = []
        start_idx = end_idx - 20

if chunk:
#     result.append(np.expand_dims(np.array(chunk), axis=0))
    result.append(np.array(chunk))
    chunk = []
    
if result:
    end_idx = image_idx+1
    save_path = os.path.join(OUT_PATH, 'chunk_{}_start_{}_end_{}.npy').format(chunk_idx, start_idx, end_idx)
    np.save(save_path, np.stack(chunks, axis=0))
    # print('>>>>>>>>>> saved to {}'.format(save_path))
    # print('>>>>>>>>>>> here', np.stack(chunks, axis=0).shape)
    chunk_idx += 1
    result = []
    start_idx = end_idx - 20
