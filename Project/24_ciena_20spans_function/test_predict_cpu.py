from predict_cpu_model import *

#here is the filename
filename = 'data_sample.csv'

#data type should be the numpy array
#shape is [batch_size, 24041] (you also can use the raw data as 24081)
data = pd.read_csv(filename, header=None).values[:50]

#use this function it will return the predict type and enlc
#shape will be [batch_size] and [batch_size, 1]
type_v, enlc_v = predict_type_enlc_cpu(data)

print
print type_v
print enlc_v

#here is the code that you can use to check the accuracy
"""
print data[:, 24050]
print data[:, 24070]
print
correct_prediction = np.equal(type_v, data[:, 24070])
print correct_prediction.astype(float)
accuracy = np.sum(correct_prediction.astype(float)) / len(data)
print accuracy
"""