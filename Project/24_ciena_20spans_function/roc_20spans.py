import os
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

prob_file_dir = '/media/freshield/COASAIR1/CIENA/Result/logs/ciena_20spans_predict_prob/probs/'
lable_file_dir = '/media/freshield/COASAIR1/CIENA/Result/logs/ciena_20spans_predict_prob/lables/'

lable_files = file_name(lable_file_dir)
files = sorted(lable_files, key=lambda d : int(d.split('pan')[-1].split('_')[0]))
lable_file_dic = {}
for i in range(20):
    lable_file_dic[i+1] = files[i]

prob_files = file_name(prob_file_dir)
files = sorted(prob_files, key=lambda d : int(d.split('pan')[-1].split('_')[0]))
prob_file_dic = {}
for i in range(20):
    prob_file_dic[i+1] = files[i]

for num, name in lable_file_dic.items():
    print num, name

for num, name in prob_file_dic.items():
    print num, name
