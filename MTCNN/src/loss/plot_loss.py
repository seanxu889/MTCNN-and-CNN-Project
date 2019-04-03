#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:12:44 2019

@author: seanxu
"""

import numpy as np
import matplotlib.pyplot as plt

f = open('onetA.txt')
data = f.readlines()
a = np.zeros((204,2))
for i in range(0,204):
    a[i][1] = float(data[i].partition('bbx: loss = ')[2].partition('. Step')[0])

for i in range(0,204):
    a[i][0] = float(data[i].partition('cls: loss = ')[2].partition('. Step')[2].partition(' for bbx')[0])

b = np.zeros((204,2))
for i in range(0,204):
    b[i][1] = float(data[i].partition('cls: loss = ')[2].partition('. Step')[0])

for i in range(0,204):
    b[i][0] = float(data[i].partition(' for cls: loss = ')[0].partition(' ')[2])

plt.plot(a[:,0],a[:,1],'b')
plt.title('O-net training proccess loss for bounding box')
plt.xlabel('step')
plt.ylabel('loss')
plt.show()

plt.plot(b[:,0],b[:,1],'g')
plt.title('O-net training proccess loss for classification')
plt.xlabel('step')
plt.ylabel('loss')
plt.show()