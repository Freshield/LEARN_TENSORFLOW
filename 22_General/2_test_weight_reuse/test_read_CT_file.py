import os
import dicom

data_path = '/home/freshield/CT'

slices = []
for s in os.listdir(data_path):
    one_slices = dicom.read_file(data_path + os.sep + s, force=True)
    slices.append(one_slices)

print(slices[0])