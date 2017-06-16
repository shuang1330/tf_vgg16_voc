import tensorflow as tf

class Network():
    def __init__(self):
        self.images = None
        self.labels = None
        self.loss = None
        self.global_variables = []
        self.predictions = {}

    def build_network(self,db):
        with tf.graph().as_default() as g:
            # build the graph
            self.images = tf.placeholder(tf.float32,shape=[None,
                                         224, 224, 3])
            self.labels = tf.placeholder(tf.int32,shape=[None,])
            self.predictions[cls_score], self.predictions[cls_prob] = \
                vgg16(images,batch_size)
            self.loss = tf.reduce_mean(
             tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=tf.reshape(cls_score, [-1, db.num_classes]),
              labels=labels))
            self.global_variables = \
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
