import os

from jcamp import JCAMP_reader
import matplotlib.pyplot as plt
import numpy as np


data_dir = './data/'
filename = 'Methane-IR_old'
fileformat = '.jdx'
output_dir = './output/'


filename_list = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith(fileformat)]

for filename in filename_list:
    jcamp_dict = JCAMP_reader(data_dir+filename+fileformat)

    print(jcamp_dict['y'].shape)

    output = np.asanyarray([10000/jcamp_dict['x'], jcamp_dict['y']])
    output = np.flip(output.transpose(), 0)

    # np.savetxt(output_dir+filename+".csv", output, delimiter=",")
