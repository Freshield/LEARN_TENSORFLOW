from predict_model import *
import os

#here is the filename
filename = 'data_sample.csv'

#data type should be the numpy array
#shape is [batch_size, 24041] (you also can use the raw data as 24081)
data = pd.read_csv(filename, header=None).values[:2]

#use this function it will return the predict type and enlc
#shape will be [batch_size] and [batch_size, 1]
model_dir_path = '/media/freshield/COASAIR1/CIENA/Result/modules/ciena_transfer_learning_train/'

target_span = 20

model_dir_path = model_dir_path + 'span%d/' % target_span

model_path = model_dir_path + 'module.ckpt'

show_accuracy = True

if os.path.exists(model_dir_path):

    type_v, enlc_v = predict_type_enlc(data,model_path)

    print
    print type_v
    print enlc_v

    # here is the code that you can use to check the accuracy
    if show_accuracy:
        print data[:, 24040 + target_span]
        print data[:, 24060 + target_span]
        print
        correct_prediction = np.equal(type_v, data[:, 24060 + target_span])
        # print correct_prediction.astype(float)
        accuracy = np.sum(correct_prediction.astype(float)) / len(data)
        print accuracy


else:
    print 'model path do not exist\n' + model_path

