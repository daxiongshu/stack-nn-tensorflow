import numpy as np
import tensorflow as tf
from util.nn_util import fully_connected
from models.base_model import BaseModel

class PGS_NET_WB(BaseModel):
    # Per-group-softmax net
    def build(self):
	print "build PGS_NET_WB" 
	params = self.params
	N = params.batch_size     # number of groups/display_ids per batch
	A = params.max_ads        # maximum number of Ads per display_id
	F = params.meta_features  # number of meta features per Ad

	yb = tf.placeholder('float32', shape=[N, A], name='yb') 	
	X = tf.placeholder('float32', shape=[N, A, F], name='x') # zero padding
	Xmask = tf.placeholder('float32', shape=[N, A], name='xmask') # in {-e10, 1}, 1 for real Ads and -e10 for padding Ads
	#Xads = tf.placeholder('float32', shape=[N], name='xads') # number of Ads per display_id
	y = tf.placeholder('float32', shape=[N, A], name='y')  # y in {0, 1} with zero padding 
	is_training = tf.placeholder(dtype=tf.bool)

	if self.params.softmax_transform:
	    print("softmax_transform")
	    Xtmp = X + tf.reshape(Xmask, [N, A, 1])
 	    Xtmp = tf.exp(Xtmp)
	    stmp = tf.reduce_sum(Xtmp, 1, keep_dims=True)+1e-5
	    Xtmp = Xtmp/stmp	
	else:
	    Xtmp = X

	with tf.name_scope("Fully-connected"):
	    Xtmp = tf.reshape(Xtmp, [N*A, F])
	    with tf.variable_scope("Layer1"):
	       	ytmp = fully_connected(Xtmp, num_neurons=50, name='W1', is_training = is_training, use_batch_norm=True, use_drop_out=False, keep_prob = 0.7, activation = 'relu', default_batch = params.default_batch)	
	    with tf.variable_scope("Layer2"):
	    	ytmp = fully_connected(ytmp, num_neurons=25, name='W2', is_training = is_training, use_batch_norm=True, use_drop_out=False, keep_prob = 1.0, activation = 'relu', default_batch = params.default_batch)
	    #with tf.variable_scope("Layer4"):
            #    ytmp = fully_connected(ytmp, num_neurons=10, name='W4', is_training = is_training, use_batch_norm=True, use_drop_out=False, keep_prob = 1.0, activation = 'relu', default_batch = params.default_batch)

	    with tf.variable_scope("Layer3"):
	    	ytmp = fully_connected(ytmp, num_neurons=1, name='W3', is_training = is_training, use_batch_norm=True, use_drop_out=False, keep_prob = 1, activation = 'None', default_batch = params.default_batch)

	# ytmp is [N*A, 1] now

	yp = tf.reshape(ytmp,[N, A])*yb + Xmask #+ yb# masking the padding Ads
	# yp is [N, A] now
	
	with tf.name_scope('Loss'):
            # Cross-Entropy loss
	    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(yp, y)
            loss = tf.reduce_mean(cross_entropy)
            total_loss = loss + params.weight_decay * tf.add_n(tf.get_collection('l2'))

	with tf.name_scope('Predict'):
	    pred = tf.nn.softmax(yp)    

	if self.params.opt == 'adam':
	    optimizer = tf.train.AdamOptimizer(params.learning_rate)
	elif self.params.opt == 'sgd':
	    optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
	elif self.params.opt == 'ada':
	    optimizer = tf.train.AdagradOptimizer(params.learning_rate)
	elif self.params.opt == 'rmsprop':
	    optimizer = tf.train.RMSPropOptimizer(params.learning_rate)
        opt_op = optimizer.minimize(total_loss, global_step=self.global_step)

	
	self.predictions = pred 	
	self.loss = cross_entropy
	self.total_loss = total_loss
	self.opt_op = opt_op

	self.yb = yb
	self.x = X
	self.y = y 
	self.xmask = Xmask
	self.is_train = is_training

    def preprocess_batch(self, batch): 
	# batch = (x, y, g, r)
	params = self.params
        N = params.batch_size     # number of groups/display_ids per batch
        A = params.max_ads        # maximum number of Ads per display_id
        F = params.meta_features  # number of meta features per Ad
	#print N,A,F
	x, y, g, r, _, b = batch
	#print "batch.r", r	
	X = np.zeros([N,A,F])
	#print "batch.x", x.shape, "X", X.shape

	Y = np.zeros([N,A])
	Xb = np.zeros([N,A])
	Xmask = np.ones([N,A])*(-1e10)
	for i in range(N):
	    if i+1 >= len(r):
		break
	    start, end = r[i], r[i+1]
	    #rtmp = range(start,end)
	    #print i, start, end, X[i,start:end,:].shape, x[start:end,:].shape, X.shape, x.shape, y.shape
	    X[i,0:end-start,:] = x[start:end,:]
	    Y[i,0:end-start] = y[start:end]
	    Xb[i,0:end-start] = b[start:end,0]
	    Xmask[i,0:end-start] = 0
	    #print rtmp, X.shape, Y.shape, y.shape 
	    #X[i,rtmp,:] = x[rtmp,:]
	    #Y[i,rtmp,:] = y[np.array(rtmp)]
	    #Xmask[i,rtmp,:] = 0

	return X, Y, Xmask,Xb

    def get_feed_dict(self, batch, is_train):
        X, Y, Xmask, Xb = self.preprocess_batch(batch)
        return {
            self.x: X,
            self.xmask: Xmask,
            self.y: Y,
            self.is_train: is_train,
	    self.yb: Xb
        }

