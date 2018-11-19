# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import glob
import json

import i3d


out_file = 'xin.json'
out = open(out_file, 'a')

TOP_N = 3
BATCH_SIZE = 8
SECOND_PER_FRAME = 1.0 / 25
_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 100
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    # 'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
tf.flags.DEFINE_string('npy_path', _SAMPLE_PATHS['rgb'], 'numpy array path')
tf.flags.DEFINE_string('npy_dir', '', 'numpy array dir')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type

  imagenet_pretrained = FLAGS.imagenet_pretrained
  npy_path = FLAGS.npy_path

  npy_dir = FLAGS.npy_dir
  if npy_dir:
    npy_list = glob.glob(os.path.join(FLAGS.npy_dir, '*.npy'))
    npy_list.sort(key=lambda x: int(os.path.splitext(x.split('_')[-1])[0]) )

    if npy_list:
      npy_dir = npy_list

  if not npy_dir:
    npy_dir = [npy_path]

  # print('>>>>>>>>>>>>> npy_dir', npy_dir)

  NUM_CLASSES = 400
  if eval_type == 'rgb600':
    NUM_CLASSES = 600

  if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

  if eval_type == 'rgb600':
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
  else:
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))


    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)


    rgb_variable_map = {}
    for variable in tf.global_variables():

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  # if eval_type in ['flow', 'joint']:
  #   # Flow input has only 2 channels.
  #   flow_input = tf.placeholder(
  #       tf.float32,
  #       shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
  #   with tf.variable_scope('Flow'):
  #     flow_model = i3d.InceptionI3d(
  #         NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
  #     flow_logits, _ = flow_model(
  #         flow_input, is_training=False, dropout_keep_prob=1.0)
  #   flow_variable_map = {}
  #   for variable in tf.global_variables():
  #     if variable.name.split('/')[0] == 'Flow':
  #       flow_variable_map[variable.name.replace(':0', '')] = variable
  #   flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb' or eval_type == 'rgb600':
    model_logits = rgb_logits
  # elif eval_type == 'flow':
  #   model_logits = flow_logits
  # else:
  #   model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)

  with tf.Session() as sess:
    feed_dict = {}


    for npy_path in npy_dir:
      items = os.path.splitext(npy_path)[0].split('_')
      chunk_idx = int(items[-5])
      start_time = int(items[-3])
      end_time = int(items[-1])

      if eval_type in ['rgb', 'rgb600', 'joint']:
        if imagenet_pretrained:
          rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        else:
          rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
        tf.logging.info('RGB checkpoint restored')
        # rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
        rgb_sample = np.load(npy_path)
        tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))      
        feed_dict[rgb_input] = rgb_sample

      # if eval_type in ['flow', 'joint']:
      #   if imagenet_pretrained:
      #     flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      #   else:
      #     flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      #   tf.logging.info('Flow checkpoint restored')
      #   flow_sample = np.load(_SAMPLE_PATHS['flow'])
      #   tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
      #   feed_dict[flow_input] = flow_sample

      out_logits_full, out_predictions_full = sess.run(
          [model_logits, model_predictions],
          feed_dict=feed_dict)

      for idx in range(len(out_logits_full)):
        out_logits = out_logits_full[idx]
        out_predictions = out_predictions_full[idx]
        sorted_indices = np.argsort(out_predictions)[::-1]
        # print('>>>>>>>>>>>>>> out_predictions', out_predictions)
        # print('>>>>>>>>>>>>>> out_logits', out_logits)

        # print('Norm of logits: %f' % np.linalg.norm(out_logits))
        # print('\nTop classes and probabilities')
        # for index in sorted_indices[:20]:
        #   print(out_predictions[index], out_logits[index], kinetics_classes[index])
        start = (start_time + idx * 80) * SECOND_PER_FRAME
        end = (start_time + idx * 80 + 100) * SECOND_PER_FRAME

        result = {'npy_path': npy_path, 'start_timestamp': start, 'end_timestamp': end, 'confidence': [], 'logit': [], 'label': [], 'result': None}
        for index in sorted_indices[:TOP_N]:
          # chunk_idx = int(os.path.splitext(npy_path)[0].split('_')[-1]) * BATCH_SIZE + idx
          # start = chunk_idx * _SAMPLE_VIDEO_FRAMES * SECOND_PER_FRAME
          # end = (chunk_idx+1) * _SAMPLE_VIDEO_FRAMES * SECOND_PER_FRAME

          # print('>>>>>>>>>>>')
          # print('>>>>>>start', start)
          # print('>>>>>>end', end)
          # print('>>>>>>>>>>>>npy_path', npy_path)
          # print('>>>>>>>>>> index', index)
          confidence, logit, label = out_predictions[index], out_logits[index], kinetics_classes[index]

          if not result['result']:
            result['result'] = label
          result['confidence'].append(str(confidence))
          result['logit'].append(str(logit))
          result['label'].append(str(label))

          print(','.join([str(item) for item in [start, end, npy_path, confidence, logit, label]]))

        json.dump(result,out)
        out.write('\n')
  out.close()

if __name__ == '__main__':
  tf.app.run(main)
