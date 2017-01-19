import tensorflow as tf
from util.data_util import DataSet,get_num_fea,write_sub
from models.pgs_net.pgs_net import PGS_NET 

flags = tf.app.flags
flags.DEFINE_string('save_dir', 'save', 'Save path [save]')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training [256]')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate [0.002]')
flags.DEFINE_integer('acc_period', 1, 'Accuracy display period [10]')
flags.DEFINE_string('task', 'cv', 'cv or test')
flags.DEFINE_string('train', 'data/train.list', 'meta list of train')
flags.DEFINE_string('display_train', '', 'display of train')
flags.DEFINE_string('test', 'data/test.list', 'meta list of test')
flags.DEFINE_string('display_test', '', 'display of test')
flags.DEFINE_string('cache', 'cache', 'cache path')
flags.DEFINE_integer('fea_limit', 5, 'Max number of features, above this value will call online code in preprocessing')
flags.DEFINE_integer('batch_size', 2, 'batch size')
flags.DEFINE_integer('max_ads', 18, 'maximum ads per display')
flags.DEFINE_integer('meta_features', 11, 'number of features')
flags.DEFINE_float('weight_decay', 0.001, 'Weight decay - 0 to turn off L2 regularization [0.001]')
flags.DEFINE_string('sub', 'sub.csv', 'path of submission')
FLAGS = flags.FLAGS

def main(_):

    FLAGS.meta_features = get_num_fea(FLAGS.train)   
    print ("number of meta features", FLAGS.meta_features)  
    train = DataSet(metalist=FLAGS.train, display=FLAGS.display_train, cache=FLAGS.cache, shuffle=True, limit = FLAGS.fea_limit) 
    test = DataSet(metalist=FLAGS.test, display=FLAGS.display_test, cache=FLAGS.cache, shuffle=False, limit = FLAGS.fea_limit)
    #train.sanity_check()
    # test's row order doesn't change!!!
    with tf.Session() as sess:
    	model = PGS_NET(FLAGS)
	sess.run(tf.initialize_all_variables())
	model.train(sess, train, test)
	preds = model.eval(sess, test, is_va = False)
	write_sub(preds,FLAGS.sub)
if __name__ == '__main__':
    tf.app.run()
    
