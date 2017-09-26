import numpy as np
import tensorflow as tf
import xlrd
import utils


#import data
data_file="heart.csv"
book=xlrd.open_workbook(data_file, encoding_override='utf-8')
sheet=book.sheet_by_index(0)
data=np.asarray([sheet.row_values(i) for i in range(1,300)])
test=np.asarray([sheet.row_values(i) for i in range(301,426)])
n_samples=sheet.nrows-1

#change string to integer
for i in range(462)
	if sheet.cell(i,4)=="present":
		sheet.cell(i,4)==1
	else:
		sheet.cell(i,4)==0
	


#define parameters
learning_rate=0.01
batch_size=100
epoch=50
#define placeholders
x=tf.placeholder(tf.float32,(batch_size,10))
y=tf.placeholder(tf.float32,(batch_size,2))

#define variables

w=tf.Variable(tf.random_normal(shape=[10,batch_sie],std_dev=0.0))
b=tf.Variable(tf.zeros(1,2))
#define y predict
logits=tf.matmul(x*w)+b
#define loss
entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
loss=tf.reduced_mean(entropy)
# Step 6: define training op
# using gradient descent to minimize loss

optimizer=tf.train.AdamOptimiser(learning_rate).minimize(loss)
with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(data.train.num_examples/batch_size)
	for i in range(epoch): # train the model n_epochs times
		total_loss = 0

		for _ in range(n_batches):
			X_batch, Y_batch = data.train.next_batch(batch_size)
			# TO-DO: run optimizer + fetch loss_batch
			_, loss_batch=sess.run([optimizer,loss],feed_dict={x=X_batch,y=Y_batch})
			# 
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	# test the model
	preds = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
	
	n_batches = int(test.test.num_examples/batch_size)
	total_correct_preds = 0
	
	for i in range(n_batches):
		X_batch, Y_batch = test.test.next_batch(batch_size)
		accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch}) 
		total_correct_preds += accuracy_batch	
	
	print('Accuracy {0}'.format(total_correct_preds/test.test.num_examples))

