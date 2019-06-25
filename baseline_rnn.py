import math

import numpy as np
import tensorflow as tf
import tflearn
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.nn.rnn_cell import GRUCell
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.nn.rnn_cell import MultiRNNCell
from tensorflow.nn.rnn_cell import DropoutWrapper, ResidualWrapper
from tensorflow.contrib.rnn import IndyGRUCell,IndyLSTMCell,GridLSTMCell,NASCell,GRUBlockCell,AttentionCellWrapper

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

class BaselineModel(object):

    def __init__(self, config):
  
        self.lab_start=config['lab_start']
        assert self.lab_start.lower() in ['true', 'false']
        if self.lab_start =='true':
            self.encoder_lab_size=config['encoder_lab_size']
        self.w2v_start=config['w2v_start']
        assert self.w2v_start.lower() in ['true', 'false']
        if self.w2v_start =='true':
            self.w2b_emb_ph = config['w2v_vector']
        
        #neuron papameter#
        self.cell_type=config['cell_type']
        assert self.cell_type.lower() in ['gru', 'indgru','indlstm','grublock','lstm']
        self.optimizer=config['optimizer']
        assert self.optimizer.lower() in ['adadelta', 'adam','rmsprop','gdsop']
        self.depth = config['depth']
        assert type(self.depth) == int
        self.embedding_size=config['embedding_size']
        assert type(self.embedding_size) == int 
        self.encoder_dx_size=config['encoder_dx_size']
        assert type(self.encoder_dx_size) == int 
        self.hidden_units=config['hidden_units']
        assert type(self.hidden_units) == int 
        self.class_number=config['class_number']
        assert type(self.class_number) == int 
        self.batch_size = config['batch_size']
        assert type(self.batch_size) == int
        
        self.use_dropout = config['use_dropout']
        assert self.use_dropout.lower() in ['true', 'false']
        if self.use_dropout.lower()=='true':
            self.dropout_rate=config['dropout_rate']
            assert self.dropout_rate<=1.0
            self.keep_prob = 1.0 - self.dropout_rate
        self.dtype=tf.float32
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=self.dtype)
    
        #loss_regulate#
        self.regulate=config['regulate']
        assert self.regulate.lower() in ['l1', 'l2','all']
        if self.regulate.lower()== 'l1':
            self.l1_regulate_rate=config['l1_regulate']
            assert type (self.l1_regulate_rate)==float
        self.regulate=config['l2_regulate']    
        if self.regulate == 'l2':
            self.l2_regulate_rate=config['l2_regulate']
            assert type (self.l2_regulate_rate)==float
        if self.regulate == 'all':
            self.l1_regulate_rate=config['l1_regulate']
            assert type (self.l1_regulate_rate)==float
            self.l2_regulate_rate=config['l2_regulate']
            assert type (self.l2_regulate_rate)==float
        self.learning_rate = config['learning_rate']
        assert type (self.learning_rate)==float
        self.max_gradient_norm = config['max_gradient_norm']
        assert type (self.max_gradient_norm)==float
        self.max_timestamp_size = config['max_timestamp_size']
        assert type(self.max_timestamp_size) == int
        self.max_variable_size = config['max_variable_size']
        assert type(self.max_variable_size) == int
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
	    tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        self.build_model()

       
    def build_model(self):
        print("building model..")

        # Building encoder and decoder networks
        self.init_placeholders()
        if self.lab_start.lower() =='true':
            self.build_lab()
        self.build_embedding()
        self.build_encoder()
        self.build_full_connect()
        self.build_loss()
        
        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()


    def init_placeholders(self):
 
        # encoder_dx_ph: [batch_size, max_time_steps]
        self.encoder_dx_ph = tf.placeholder(tf.int32, [self.batch_size,self.max_timestamp_size,self.max_variable_size], name='encoder_inputs_ph')
        #final_calss#
        self.target_ph_class=tf.placeholder(tf.float32,[None,None],name='target_ph_class')
        #encodr_length#
        self.en_seq_len_ph=tf.placeholder(tf.int32,[None],name='en_seq_len_ph')
        #decoder_length#
        self.class_weight_ph=tf.placeholder(tf.float32,[None],name='class_weight_ph')
        #dropout_rate#
        self.keep_prob_ph=tf.placeholder(tf.float32,name='dropout_rate_ph')
            
        if self.lab_start =='true':
            # encoder_value_inputs: [batch_size, max_time_steps,value]
            self.encoder_lab_ph = tf.placeholder(tf.float32,[None,None,None], name='encoder_lab_ph')
    def build_lab(self):
        with tf.variable_scope('lab_process'):
            tf.get_variable_scope().reuse_variables()
            self.lab_cell = self.build_cell_layer()
            self.lab_inputs= tf.layers.dense(self.encoder_lab_ph,self.hidden_units, dtype=self.dtype, name='input_projection',kernel_initializer=self.initializer)
            self.lab_outputs,_= tf.nn.dynamic_rnn(cell=self.lab_cell, inputs=self.lab_inputs,sequence_length=self.en_seq_len_ph, dtype=self.dtype,time_major=False,parallel_iterations=8)
            
    def build_embedding(self):
        print("build embedding")
        if self.w2v_start =='true':
            with tf.variable_scope('embedding',reuse=tf.AUTO_REUSE):
                mask_array=[[1.]]*0+[[0.]]+[[1.]]*(self.encoder_dx_size-0-1)
                mask_padding_lookup_table=tf.get_variable(name='maskembedding', dtype=self.dtype,trainable=False,initializer=mask_array)
                initval=tf.constant_initializer(np.array(self.w2b_emb_ph))
                w=tf.get_variable(name='embedding',shape=[self.encoder_dx_size, self.embedding_size], dtype=self.dtype,trainable=True,initializer=initval)
                #w=w.assign(self.w2b_emb_ph)
                self.encoder_embeddings=w
                self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.encoder_embeddings, ids=self.encoder_dx_ph)
                self.encoder_inputs_mask= tf.nn.embedding_lookup(params=mask_padding_lookup_table, ids=self.encoder_dx_ph)
                self.encoder_inputs_embedded = self.encoder_inputs_embedded*self.encoder_inputs_mask
                self.encoder_inputs_embedded = tf.squeeze (self.encoder_inputs_embedded)
                self.encoder_inputs_embedded =tf.reduce_sum(self.encoder_inputs_embedded,2)
                self.encoder_inputs_embedded1=self.encoder_inputs_embedded
                #self.encoder_inputs_embedded_size=self.encoder_inputs_embedded.shape[2].value
                self.encoder_inputs_embedded_weight=tf.get_variable(name='full_emb_weight',shape=[self.embedding_size, self.hidden_units],initializer=self.initializer, dtype=self.dtype,trainable=True)
                
                self.encoder_inputs_embedded =tf.tensordot(self.encoder_inputs_embedded,self.encoder_inputs_embedded_weight,axes=1)
                print (self.encoder_inputs_embedded)
                self.encoder_inputs_embedded = tf.squeeze (self.encoder_inputs_embedded)
                self.en_emb=self.encoder_inputs_embedded
        else:
            with tf.variable_scope('embedding',reuse=tf.AUTO_REUSE):
                mask_array=[[1.]]*0+[[0.]]+[[1.]]*(self.encoder_dx_size-0-1)
                mask_padding_lookup_table=tf.get_variable(name='maskembedding', dtype=self.dtype,trainable=False,initializer=mask_array)
                self.encoder_embeddings = tf.get_variable(name='embedding',shape=[self.encoder_dx_size, self.embedding_size], dtype=self.dtype,trainable=True,initializer=self.initializer)
                self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.encoder_embeddings, ids=self.encoder_dx_ph)
                self.encoder_inputs_mask= tf.nn.embedding_lookup(params=mask_padding_lookup_table, ids=self.encoder_dx_ph)
                self.encoder_inputs_embedded = self.encoder_inputs_embedded*self.encoder_inputs_mask
                self.encoder_inputs_embedded = tf.squeeze (self.encoder_inputs_embedded)
                self.encoder_inputs_embedded =tf.reduce_sum(self.encoder_inputs_embedded,2)
                self.encoder_inputs_embedded1=self.encoder_inputs_embedded
                #self.encoder_inputs_embedded_size=self.encoder_inputs_embedded.shape[2].value
                #self.encoder_inputs_embedded_weight=tf.get_variable(name='full_emb_weight',shape=[self.encoder_inputs_embedded_size, self.hidden_units],initializer=self.initializer, dtype=self.dtype,trainable=True)
                self.encoder_inputs_embedded_weight=tf.get_variable(name='full_emb_weight',shape=[self.embedding_size, self.hidden_units],initializer=self.initializer, dtype=self.dtype,trainable=True)
                self.encoder_inputs_embedded =tf.tensordot(self.encoder_inputs_embedded,self.encoder_inputs_embedded_weight,axes=1)
                self.encoder_inputs_embedded = tf.squeeze (self.encoder_inputs_embedded)
                self.en_emb=self.encoder_inputs_embedded
    def encoder_simple(self):
        with tf.variable_scope('encoder_simple',reuse=tf.AUTO_REUSE):
            encoder_cell=self.build_cell_layer()
            encoder_outputs, encoder_last_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=self.en_emb,sequence_length=self.en_seq_len_ph, dtype=self.dtype,time_major=False)
            return encoder_outputs, encoder_last_state            
    
    def build_encoder(self):
        print("building encoder..")
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            # Building encoder_cell
            encoder_cell=self.build_cell_layer()
            if self.lab_start =='true':
                labresult= self.lab_outputs
                self.en_emb1=self.en_emb
                self.en_emb1=tf.concat([self.en_emb1,labresult],2)
                self.en_emb1= tf.layers.dense(self.en_emb1,self.hidden_units, dtype=self.dtype, name='embdense',kernel_initializer=self.initializer)
            else:
                self.en_emb1=self.en_emb
            self.encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=self.en_emb1,sequence_length=self.en_seq_len_ph, dtype=self.dtype,time_major=False)
            #print (self.en_emb1)
            #print (self.en_seq_len_ph)
                       
            #self.encoder_outputs, _ = self.encoder_simple()
         
    
    def build_full_connect(self):
        print("building full_ct..")
        with tf.variable_scope('full_connect',reuse=tf.AUTO_REUSE):
            self.c_t=tf.reshape(self.encoder_outputs,[self.batch_size,-1])
            self.c_t_size=self.c_t.shape[1].value
            w_output=tf.get_variable(name='full_ct_class_weight',shape=[self.c_t_size, self.class_number],initializer=self.initializer, dtype=self.dtype,trainable=True)
            b_output=tf.get_variable(name='full_ct_class_bias',shape=[self.class_number],initializer=tf.zeros_initializer(), dtype=self.dtype,trainable=True)
            self.y_hat=tf.nn.xw_plus_b(self.c_t,w_output,b_output)
            self.y_hat1=tf.nn.softmax(self.y_hat)
        
    def build_loss(self):
         #masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,maxlen=max_decoder_length, dtype=self.dtype, name='masks')
        print("building loss..")
        with tf.variable_scope('loss',reuse=tf.AUTO_REUSE):          
            self.class_loss=tf.losses.softmax_cross_entropy(onehot_labels=self.target_ph_class,logits=self.y_hat1,weights=self.class_weight_ph)
            
            
            self.total_loss=self.class_loss
            if self.regulate== 'l1':
                varsall=tf.trainable_variables()
                self.l1_loss=tf.add_n([tflearn.losses.L1(v,self.l1_regulate_rate) for v in varsall])
                self.total_loss=self.total_loss+self.l1_loss
            elif self.regulate== 'l2':
                varsall=tf.trainable_variables()
                self.l2_loss=tf.add_n([tflearn.losses.L2(v,self.l2_regulate_rate) for v in varsall])
                self.total_loss=self.total_loss+self.l2_loss
            elif self.regulate== 'all':
                varsall=tf.trainable_variables()
                self.l1_loss=tf.add_n([tflearn.losses.L1(v,self.l1_regulate_rate) for v in varsall])
                self.l2_loss=tf.add_n([tflearn.losses.L2(v,self.l2_regulate_rate) for v in varsall])
                self.total_loss=self.total_loss+self.l1_loss+self.l2_loss
            else:
                self.total_loss
            
            self.init_optimizer()    
    def build_single_cell(self):
   
        if (self.cell_type.lower() == 'gru'):
            cell_type = GRUCell
        elif (self.cell_type.lower() == 'indgru'):
            cell_type = IndyGRUCell
        elif (self.cell_type.lower() == 'indlstm'):
            cell_type = IndyLSTMCell
        elif (self.cell_type.lower() == 'grublock'):
            cell_type = GRUBlockCell
        elif self.cell_type.lower() == 'lstm':
            cell_type = LSTMCell 
        if self.cell_type.lower() == 'lstm':
            cell=cell_type(self.hidden_units,initializer=self.initializer)
        elif self.cell_type.lower() =='grublock':
            cell=cell_type(self.hidden_units)
        else:
            cell = cell_type(self.hidden_units,kernel_initializer=self.initializer,bias_initializer=tf.zeros_initializer)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,output_keep_prob=self.keep_prob_ph)
        return cell
    
    def build_cell_layer (self):
        building_cell=self.build_single_cell()
        return MultiRNNCell([building_cell for i in range(self.depth)])
   
    def init_optimizer(self):
        print("setting optimizer..")
        with tf.variable_scope('loss',reuse=tf.AUTO_REUSE):
            trainable_params = tf.trainable_variables()
            if self.optimizer.lower() == 'adadelta':
                self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer.lower() == 'adam':
                self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer.lower() == 'rmsprop':
                self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer.lower() == 'gdsop':
                self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            #self.updates = self.opt.minimize(self.total_loss)

        # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.total_loss, trainable_params)

        # Clip gradients by a given maximum_gradient_norm
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Update the model
            self.updates = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)
    def fed_dict (self,batch_dict,lab=None,w2v=None):
        class_weight=batch_dict['classweight']
        target_ph_class=batch_dict['patientclass']
        encoder_dx_ph=batch_dict['newseqs']
        en_seq_len_ph=batch_dict['timestamp']
        if self.lab_start=='true' and self.w2v_start =='true':
            feed_dict={}
            feed_dict[self.encoder_dx_ph]=encoder_dx_ph
            feed_dict[self.target_ph_class]=target_ph_class
            feed_dict[self.en_seq_len_ph]=en_seq_len_ph
            feed_dict[self.class_weight_ph]=class_weight
            feed_dict[self.encoder_lab_inputs]=lab
            
        elif self.lab_start=='true' and self.w2v_start =='false':
            feed_dict={}
            feed_dict[self.encoder_dx_ph]=encoder_dx_ph
            feed_dict[self.target_ph_class]=target_ph_class
            feed_dict[self.en_seq_len_ph]=en_seq_len_ph
            feed_dict[self.class_weight_ph]=class_weight
            feed_dict[self.encoder_lab_inputs]=lab
        elif self.lab_start=='false' and self.w2v_start =='true':
            feed_dict={}
            feed_dict[self.encoder_dx_ph]=encoder_dx_ph
            feed_dict[self.target_ph_class]=target_ph_class
            feed_dict[self.en_seq_len_ph]=en_seq_len_ph
            feed_dict[self.class_weight_ph]=class_weight
            
        else:
            feed_dict={}
            feed_dict[self.encoder_dx_ph]=encoder_dx_ph
            feed_dict[self.target_ph_class]=target_ph_class
            feed_dict[self.en_seq_len_ph]=en_seq_len_ph
            feed_dict[self.class_weight_ph]=class_weight
        return feed_dict
                      
    def train(self, sess,feed_dict):
        
        input_feed = feed_dict
        # Input feeds for dropout
        input_feed[self.keep_prob_ph] = self.keep_prob
 
        output_feed = [self.total_loss,self.updates]
        
        outputs = sess.run(output_feed, input_feed)
        return outputs[0],outputs[1]	# loss, summary


    def evalm(self, sess, feed_dict):
        
        input_feed = feed_dict
        # Input feeds for dropout
        input_feed[self.keep_prob_ph] = 1.0

        output_feed = [self.y_hat1]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]


    def predict(self, sess, feed_dict):
        
        input_feed = feed_dict

        # Input feeds for dropout
        input_feed[self.keep_prob_ph] = 1.0
 
        output_feed = [self.y_hat1]
        outputs = sess.run(output_feed, input_feed)
        
        return outputs[0]