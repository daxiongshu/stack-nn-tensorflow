import tensorflow as tf
from tqdm import tqdm
from termcolor import colored
import numpy as np
from util.evaluate import apk12

class BaseModel(object):

    def __init__(self, params):
	self.params = params
	self.save_dir = params.save_dir
	
	with tf.variable_scope("Stack_NN"):
	    #print("building stack nn...")
	    self.global_step = tf.Variable(0, name='global_step', trainable=False)
	    self.build()

    ###############################################
    # Start: virtual functions to be implemented
    ###############################################
    def build(self):
	#pass
	raise NotImplementedError()

    def get_feed_dict(self, batch, is_train):
        raise NotImplementedError()

    def preprocess_batch(self, batch):
	raise NotImplementedError()
    ###############################################
    # End: virtual functions to be implemented
    ###############################################


    ###############################################
    # Start: common functions to be inherited
    ###############################################

    def train_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, is_train=True)
        return sess.run([self.opt_op, self.global_step], feed_dict=feed_dict)

    def test_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, is_train=False)
        return sess.run([self.loss,  self.global_step, self.predictions], feed_dict=feed_dict)

    def train(self, sess, train_data, val_data=None):
	params = self.params
        num_epochs = params.num_epochs
        num_batches = (train_data.num_groups + self.params.batch_size -1)/self.params.batch_size

	print("Training %d epochs ..." % num_epochs)
	for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
	    losses = []
	    #for i in range(num_batches):
	    while True:
		batch = train_data.next_batch(self.params.batch_size) # random shuffled batch
		self.train_batch(sess, batch)
		if batch[4]:
		    break
		#losses.append(loss)
	    #train_data.reset()

	    if (epoch_no + 1) % params.acc_period == 0:
		print()  # Newline for TQDM
		#print("[Train] step %d: Loss = %.4f" % \
              	#( global_step, np.mean(losses)))
		if val_data:
		    self.eval(sess, val_data, is_va = True)	
	    train_data.reset()		    

    def eval(self, sess, data, is_va = False):
	data.reset()
	num_batches = (data.num_groups + self.params.batch_size -1)/self.params.batch_size 
	name = 'Validation' if is_va  else 'Test'
	apk_results = []
	predictions = []
	losses = []
	#for _ in range(num_batches):
	while True:
	    batch = data.next_batch(self.params.batch_size) # continuous batch
	    # batch is a tuple (X, y, dispaly_id)
	    loss, global_step, prediction = self.test_batch(sess, batch)
	    apk_result, ypred = apk12(batch, prediction)
	    apk_results.append(apk_result)
	    predictions.append(ypred)
	    losses.append(loss)
	    if batch[4]:
		break
	print(colored("[%s] step %d: APK-12 = %.4f, Loss = %.4f"%(name, global_step, np.mean(apk_results), np.mean(losses)), 'green'))
	#data.reset()
	return np.concatenate(predictions) # row order is the same as input test

    def save(self, sess):
        print("Saving model to %s" % self.save_dir)
        self.saver.save(sess, self.save_dir, self.global_step)

    def load(self, sess):
        print("Loading model ...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)


    ###############################################
    # End: common functions to be inherited
    ###############################################
