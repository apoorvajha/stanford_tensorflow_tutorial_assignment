from process_data import process_data
import tensorflow as tf
import numpy as np


vocab_size=10000
batch_size=300
skip_window=10

def coocurrance_embeddings(batch):
	with tf.namescope('data')
		centre_words=tf.placeholder(tf.int32,shape(batch_size))
		context_word=tf.placeholder(tf.int32,shape(batch_size,1))
	with tf.name_scope('embeddings')
		embeddings=tf.zeros([batch_size,batch_size])		














def main():
	batch=process_data(vocab_size,batch_size,skip_window)
	coocurrance_embeddings(batch)
