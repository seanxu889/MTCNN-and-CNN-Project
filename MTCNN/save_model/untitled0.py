#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:15:43 2019

@author: seanxu
"""

from tensorflow.python import pywrap_tensorflow
import numpy as np

checkpoint_path="/Users/seanxu/Desktop/Learn/Python/Spyder_py3/xss/4_MTCNN-Tensorflow-master/save_model/trymy/pnet/checkpoint"
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()
param =[]

for key in var_to_shape_map:
    print ("tensor_name",key)
    param.append(reader.get_tensor(key))

np.save('p111.npy',param)
