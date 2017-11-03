from predict_cpu_model import *
import os

#here is the filename
filename = 'data_sample.csv'

#data type should be the numpy array
#shape is [batch_size, 24041] (you also can use the raw data as 24081)
data = pd.read_csv(filename, header=None).values[:2]

target_spans = [i + 1 for i in range(20)]

print_values = False
show_accuracy = True

results = {}

for i in target_spans:

    # use this function it will return the predict type and enlc
    # shape will be [batch_size] and [batch_size, 1]
    model_dir_path = '/media/freshield/COASAIR1/CIENA/Result/modules/ciena_transfer_learning_train/'

    target_span = i

    model_dir_path = model_dir_path + 'span%d/' % target_span

    model_path = model_dir_path + 'module.ckpt'
    print
    print '=============================Span%d test now==============================\n' % target_span

    if os.path.exists(model_dir_path):

        type_v, enlc_v = predict_type_enlc_cpu(data, model_path)

        results['span%d_type'%target_span] = type_v
        results['span%d_enlc'%target_span] = enlc_v


        if print_values:
            print 'Predict Type'
            print type_v
            print
            print 'Predict ENLC'
            print enlc_v

        # here is the code that you can use to check the accuracy
        if show_accuracy:
            if print_values:
                print
                print 'True Type'
                print data[:, 24060 + target_span]
                print
                print 'True ENLC'
                print data[:, 24040 + target_span]
                print
            print 'Accuracy'
            correct_prediction = np.equal(type_v, data[:, 24060 + target_span])
            # print correct_prediction.astype(float)
            accuracy = np.sum(correct_prediction.astype(float)) / len(data)
            print accuracy


    else:
        print '\nSPAN%d model not exist\n' % target_span + model_path

