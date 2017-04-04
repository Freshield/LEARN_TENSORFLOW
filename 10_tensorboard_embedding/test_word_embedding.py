from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import os

#parameter
N = 2000
D = 200
LOG_DIR = 'word_logs'

#input
embedding_var = tf.Variable(tf.random_normal([N, D]),trainable=False, name='word_embedding')

#saver
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), N)
print sess.run(embedding_var)

#config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
#embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

