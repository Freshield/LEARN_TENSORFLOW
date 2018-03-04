import os
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

file_dir = '/media/freshield/COASAIR1/CIENA/Result/logs/all_20spans_predict_data/'

files = file_name(file_dir)

files = sorted(files, key=lambda d : int(d.split('pan')[-1].split('_')[0]))
print files

dic = {}
for i in range(20):
    dic[i+1] = files[i]

print dic