import sys
 
if '../' not in sys.path: sys.path.append('../')
import time
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import data_utils
from utils import eval_utils
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from tensorflow.python.layers.core import Dense
from varAttention_decoder import basic_decoder
from varAttention_decoder import decoder
from varAttention_decoder import attention_wrapper
import random
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
from inferenceHelper import *

class VarSeq2SeqVarAttnModel(object):

    def __init__(self, config, encoder_embeddings_matrix, decoder_embeddings_matrix):#,
                 #encoder_word_index, decoder_word_index):

        self.config = config

        self.lstm_hidden_units = config['lstm_hidden_units']
        self.embedding_size = config['embedding_size']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']

        self.encoder_vocab_size = encoder_embeddings_matrix.shape[1]#encoder_embeddings_matrix.shape[1] #config['encoder_vocab']
        self.decoder_vocab_size = 1 #config['decoder_vocab']

        self.encoder_num_tokens = config['encoder_num_tokens']
        self.decoder_num_tokens = config['decoder_num_tokens']

        self.dropout_keep_prob = config['dropout_keep_prob']
        self.word_dropout_keep_probability = config['word_dropout_keep_probability']
        self.z_temp = config['z_temp']
        self.attention_temp = config['attention_temp']
        self.use_hmean = config['use_hmean']
        self.gamma_val = config['gamma_val']

        self.initial_learning_rate = config['initial_learning_rate']
        self.learning_rate_decay = config['learning_rate_decay']
        self.min_learning_rate = config['min_learning_rate']

        self.batch_size = config['batch_size']
        self.epochs = config['n_epochs']

        self.encoder_embeddings_matrix = encoder_embeddings_matrix
        self.decoder_embeddings_matrix = decoder_embeddings_matrix
        #self.encoder_word_index = encoder_word_index
        #self.decoder_word_index = decoder_word_index
        #self.encoder_idx_word = dict((i, word) for word, i in encoder_word_index.items())
        #self.decoder_idx_word = dict((i, word) for word, i in decoder_word_index.items())

        self.logs_dir = config['logs_dir']
        self.model_checkpoint_dir = config['model_checkpoint_dir']
        self.bleu_path = config['bleu_path']

        #self.pad = self.decoder_word_index['PAD']
        #self.eos = self.decoder_word_index['EOS']

        self.epoch_bleu_score_val = {'1': [], '2': [], '3': [], '4': []}
        self.log_str = []

        self.build_model()

    def build_model(self):
        print("[INFO] Building Model ...")

        self.init_placeholders()
        self.embedding_layer()
        self.build_encoder()
        self.build_latent_space()
        self.build_decoder()
        self.loss()
        self.optimize()
        self.summary()

    def init_placeholders(self):
        with tf.name_scope("model_inputs"):
            # Create palceholders for inputs to the model
            #FIXME int32 ?
            self.input_data = tf.placeholder(tf.float32, [self.batch_size, self.encoder_embeddings_matrix.shape[1]], name='input') #
            self.target_data = tf.placeholder(tf.float32, [self.batch_size, 1], name='targets') #self.decoder_embeddings_matrix[1]
            self.lr = tf.placeholder(tf.float32, name='learning_rate', shape=())
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # Dropout Keep Probability
            #self.source_sentence_length = tf.placeholder(tf.int32, shape=(self.batch_size,),
            #                                             name='source_sentence_length')
            self.target_sentence_length = tf.placeholder(tf.int32, shape=(self.batch_size,),
                                                         name='target_sentence_length')
            self.word_dropout_keep_prob = tf.placeholder(tf.float32, name='word_drop_keep_prob', shape=())
            self.lambda_coeff = tf.placeholder(tf.float32, name='lambda_coeff', shape=())
            self.gamma_coeff = tf.placeholder(tf.float32, name='gamma_coeff', shape=())
            self.z_temperature = tf.placeholder(tf.float32, name='z_temperature', shape=())
            self.attention_temperature = tf.placeholder(tf.float32, name='attention_temperature', shape=())

    def embedding_layer(self):
        with tf.name_scope("word_embeddings"):
            self.encoder_embeddings = tf.Variable(
                initial_value=np.array(self.encoder_embeddings_matrix, dtype=np.float32),
                dtype=tf.float32, trainable=False)
            #self.enc_embed_input = tf.nn.embedding_lookup(self.encoder_embeddings, self.input_data)
            self.enc_embed_input =tf.reshape(self.input_data,shape=[-1,1,self.encoder_embeddings_matrix.shape[1]])#s, dtype=tf.float32)#tf.nn.embedding_lookup(self.decoder_embeddings, self.dec_input)
                
            print(f'self.enc_embed_input {self.enc_embed_input} and  shape {self.enc_embed_input.shape}')
            # self.enc_embed_input = tf.nn.dropout(self.enc_embed_input, keep_prob=self.keep_prob)

            with tf.name_scope("decoder_inputs"):
                self.decoder_embeddings = tf.Variable(
                    initial_value=np.array(self.decoder_embeddings_matrix, dtype=np.float32),
                    dtype=tf.float32, trainable=False)
                
                keep = tf.where(
                    tf.random_uniform([self.batch_size, 1]) < self.word_dropout_keep_prob, # FIXME
                    tf.fill([self.batch_size, 1], True), #self.encoder_embeddings_matrix.shape[1]
                    tf.fill([self.batch_size, 1], False)) #self.encoder_embeddings_matrix.shape[1]

                print(f'keep {keep} and shape {keep.shape}')
                #FIXME instead of 1 --> self.encoder_embeddings_matrix.shape[1]
                # FIXME tf.int32 for self.target_data too
                ending = tf.cast(keep, dtype=tf.float32) * self.target_data
                #ending = tf.strided_slice(ending, [0,0], [self.batch_size, -1], [1,1], #2dim
                #                          name='slice_input')  # Minus 1 implies everything till the last dim
                print(f'ending {ending} with shape {ending.shape}')
                # self.dec_input = tf.concat([tf.fill([self.batch_size, 1], self.decoder_word_index['GO']), ending], 1,
                #                            name='dec_input')
                #FIXME is that correct ? check without embedding
                self.dec_input = ending #tf.concat([tf.fill([self.batch_size, 1], 2), ending], 1,name='dec_input')
                print(f'self.dec_input {self.dec_input} and  shape {self.dec_input.shape}')
                self.dec_embed_input = tf.cast(tf.reshape(self.dec_input,shape=[-1,1,1]), dtype=tf.float32)#tf.nn.embedding_lookup(self.decoder_embeddings, self.dec_input)
                #self.dec_embed_input = tf.reshape(self.dec_embed_input,shape=[-1])
                print(f'dec_embed_input {self.dec_embed_input} and shape {self.dec_embed_input.shape}')
                # self.dec_embed_input = tf.nn.dropout(self.dec_embed_input, keep_prob=self.keep_prob)

    def build_encoder(self):
        with tf.name_scope("encode"):
            for layer in range(self.num_layers):
                with tf.variable_scope('encoder_{}'.format(layer + 1)):
                    cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(self.lstm_hidden_units)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob)

                    cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(self.lstm_hidden_units)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob)

                    self.enc_output, self.enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                      cell_bw,
                                                                                      self.enc_embed_input,
                                                                                      #self.source_sentence_length,
                                                                                      dtype=tf.float32)

            # Join outputs since we are using a bidirectional RNN
            self.h_N = tf.concat([self.enc_state[0][1], self.enc_state[1][1]], axis=-1,
                                 name='h_N')  # Concatenated h from the fw and bw LSTMs
            self.enc_outputs = tf.concat([self.enc_output[0], self.enc_output[1]], axis=-1, name='encoder_outputs')
            print(f'self.enc_outputs {self.enc_outputs.shape}')

    def build_latent_space(self):
        with tf.name_scope("latent_space"):
            self.z_mean = Dense(self.latent_dim, name='z_mean')(self.h_N)
            self.z_log_sigma = Dense(self.latent_dim, name='z_log_sigma')(self.h_N)

            self.z_vector = tf.identity(self.sample_gaussian(), name='z_vector')

    def sample_gaussian(self):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope('sample_gaussian'):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(self.z_log_sigma), name='epsilon')
            return self.z_mean + tf.scalar_mul(self.z_temperature,
                                               epsilon * tf.exp(self.z_log_sigma))  # N(mu, I * sigma**2)

    def calculate_kl_loss(self):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            return -0.5 * tf.reduce_sum(1.0 + 2 * self.z_log_sigma - self.z_mean ** 2 -
                                        tf.exp(2 * self.z_log_sigma), 1)

    def build_decoder(self):
        with tf.variable_scope("decode"):
            for layer in range(self.num_layers):
                with tf.variable_scope('decoder_{}'.format(layer + 1)):
                    dec_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(2 * self.lstm_hidden_units)
                    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=self.keep_prob)
            #FIXME
            self.output_layer = Dense(1)

            attn_mech = attention_wrapper.LuongAttention(2 * self.lstm_hidden_units,
                                                          self.enc_outputs)#,
                                                          #memory_sequence_length=self.source_sentence_length)

            attn_cell = attention_wrapper.AttentionWrapper(dec_cell, attn_mech, self.attention_temperature, self.use_hmean, self.lstm_hidden_units)

            self.init_state = attn_cell.zero_state(self.batch_size, tf.float32)

            with tf.name_scope("training_decoder"):
                print('self.dec_embed_input',self.dec_embed_input)
                print('self.target_sentence_length',self.target_sentence_length)
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.dec_embed_input,
                                                                    sequence_length=self.target_sentence_length,
                                                                    time_major=False)
                #print(f"training_helper {training_helper.initialize()}")
                training_decoder = basic_decoder.BasicDecoder(attn_cell, 
                                                              training_helper,
                                                              initial_state=self.init_state,
                                                              latent_vector=self.z_vector,
                                                              output_layer=self.output_layer)

                #print(f"training_helper {tf.shape(training_helper)},training_decoder {tf.shape(training_decoder)}")
                self.training_logits, _state, _len, self.c_kl_batch_train = decoder.dynamic_decode(training_decoder,
                                                                                                              output_time_major=False,
                                                                                                              impute_finished=True,
                                                                                                              maximum_iterations=1,
                                                                                                              name='dec') #FIXME self.decoder_num_tokens
                #FIXME tf.identity
                self.training_logits = tf.identity(self.training_logits.rnn_output, 'logits')

            with tf.name_scope("inference_decoder"):
                #start_token = self.decoder_word_index['GO']
                #end_token = self.decoder_word_index['EOS']
                start_token = -1. #self.decoder_word_index['GO']
                end_token = 0 #self.decoder_word_index['EOS']

                
                #start_tokens = tf.constant(start_token, shape=[self.batch_size, 1])
                start_tokens = tf.tile([start_token], [self.batch_size],
                                       name='start_tokens')
                #start_tokens = tf.reshape(start_tokens,shape=[self.batch_size,1])
                start_tokens = tf.reshape(start_tokens,shape=[-1,1])
                print("start_tokens", start_tokens, 'and shape', start_tokens.shape[0])
                from tensorflow.python.ops import array_ops
                print("shape start_tokens", array_ops.shape(start_tokens)[0])
                                   
                # This is an inference helper without embedding. The sample_ids are the
                # actual output in this case (not dealing with any logits here).
                # The end_fn is always False because the data is provided by a generator
                # that will stop once it reaches output_size. This could be
                # extended to outputs of various size if we append end tokens, and have
                # the end_fn check if sample_id return True for an end token.
                # FIXME tf.contrib.seq2seq.
                def sample_fn(outputs):
                    # from tensorflow.python.ops import math_ops
                    # from tensorflow.python.framework import dtypes
                    # sample_ids = math_ops.cast(math_ops.argmax(outputs, axis=-1), dtypes.int32)
                    return tf.reshape(outputs,[-1])

                def end_fn(sample_ids):
                    return tf.tile([False], [self.batch_size])
                
                # def next_inputs_fn(sample_ids):
                #     return tf.reshape(tf.cast(sample_ids,tf.int32),[-1])

                inference_helper = tf.contrib.seq2seq.InferenceHelper(
                    sample_fn=sample_fn,#lambda outputs: outputs
                    sample_shape=[1], #self.batch_size
                    sample_dtype=tf.float32,
                    start_inputs=start_tokens,
                    end_fn=end_fn) #lambda sample_ids: False
                    #,next_inputs_fn=next_inputs_fn 
                #print("inf batch", inference_helper._batch_size)
                inference_helper._batch_size = self.batch_size
                print('inference_helper.ini',inference_helper.initialize()[0])
                # inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.decoder_embeddings,
                #                                                             start_tokens,
                #                                                             end_token)
                
                inference_decoder = basic_decoder.BasicDecoder(attn_cell,
                                                               inference_helper,
                                                               initial_state=self.init_state,
                                                               latent_vector=self.z_vector,
                                                               output_layer=self.output_layer)
                print('inference_decoder',inference_decoder)
                self.inference_logits, _state, _len, self.c_kl_batch_inf  = decoder.dynamic_decode(inference_decoder,
                                                                                                              output_time_major=False,
                                                                                                              impute_finished=True, #True
                                                                                                              maximum_iterations=1) #self.decoder_num_tokens

                self.inference_logits = tf.identity(self.inference_logits.rnn_output, name='predictions')

                self.c_kl_batch_train = tf.div(self.c_kl_batch_train, tf.cast(self.target_sentence_length,
                                                                         dtype=tf.float32))  # Divide by respective target seq lengths

    
    def correlation_coefficient(self):
        target_data = tf.reshape(self.target_data,shape=[-1])
        pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(self.training_logits,target_data)
        # find all variables created for this metric
        metric_vars = [i for i in tf.local_variables() if 'correlation_coefficient' in i.name.split('/')[1]]

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with tf.control_dependencies([update_op]):
            pearson_r = tf.identity(pearson_r)
            return 1-pearson_r**2

    def loss(self):
        with tf.name_scope('losses'):
            self.kl_loss = self.calculate_kl_loss()
            self.kl_loss = tf.scalar_mul(self.lambda_coeff, self.kl_loss)

            self.context_kl_loss = tf.scalar_mul(self.gamma_coeff * self.lambda_coeff, self.c_kl_batch_train)

            #batch_maxlen = tf.reduce_max(self.target_sentence_length)
            
            # the training decoder only emits outputs equal in time-steps to the
            # max time in the current batch
            # target_sequence = tf.slice(
            #     input_=self.target_data,
            #     begin=[0, 0],
            #     size=[self.batch_size, batch_maxlen],
            #     name="target_sequence")

            # Create the weights for sequence_loss
            # masks = tf.sequence_mask(self.target_sentence_length, batch_maxlen, dtype=tf.float32, name='masks')

            # self.xent_loss = tf.contrib.seq2seq.sequence_loss(
            #     self.training_logits,
            #     target_sequence,
            #     weights=masks,
            #     average_across_batch=False)

            self.training_logits = tf.reshape(self.training_logits,shape=[-1])
            target_data = tf.reshape(self.target_data,shape=[-1])
            #self.xent_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_data,logits=self.training_logits) #softmax
            self.xent_loss = tf.pow(tf.clip_by_value(target_data - self.training_logits,-1.,1.),2)
            

            # pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(self.training_logits, target_data, name='pearson_r')
            # # metric_vars = [i for i in tf.local_variables() if 'pearson_r'  in i.name.split('/')]
            # # # Add metric variables to GLOBAL_VARIABLES collection.
            # # # They will be initialized for new session.
            # # print('pearson_r',pearson_r)
            # # for v in metric_vars:
            # #     tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
            # #pearson_r = tf.identity(pearson_r)
            # print('update_op',update_op)
            # self.xent_loss2 =  tf.constant(1.) -pearson_r**2
            #self.xent_loss2 = self.correlation_coefficient()

            print('YO   self.xent_loss',self.xent_loss)
            #print('YO   self.xent_loss2',self.xent_loss2)
            #self.xent_loss = tf.keras.losses.mean_squared_error(self.training_logits,self.target_data)
            # L2-Regularization
            self.var_list = tf.trainable_variables()
            self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.var_list if 'bias' not in v.name]) * 0.001
            print('YO   self.lossL2',self.lossL2)
            self.cost = tf.reduce_sum(self.xent_loss + self.kl_loss + self.context_kl_loss) + self.lossL2  #+ tf.abs(self.xent_loss2) * 200


    def optimize(self):
        # Optimizer
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(self.lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost, var_list=self.var_list)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)

    def summary(self):
        with tf.name_scope('summaries'):
            #tf.summary.scalar('xent_loss2', tf.reduce_sum(self.xent_loss2))
            tf.summary.scalar('xent_loss', tf.reduce_sum(self.xent_loss))
            tf.summary.scalar('l2_loss', tf.reduce_sum(self.lossL2))
            tf.summary.scalar("kl_loss", tf.reduce_sum(self.kl_loss))
            tf.summary.scalar("context_kl_loss", tf.reduce_sum(self.context_kl_loss))
            tf.summary.scalar('total_loss', tf.reduce_sum(self.cost))            
            tf.summary.histogram("latent_vector", self.z_vector)
            tf.summary.histogram("latent_mean", self.z_mean)
            tf.summary.histogram("latent_log_sigma", self.z_log_sigma)
            self.summary_op = tf.summary.merge_all()

    def train(self, x_train, y_train, x_val, y_val, true_val):

        print('[INFO] Training process started')

        learning_rate = self.initial_learning_rate
        iter_i = 0
        lambda_val = 0.0

        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            writer = tf.summary.FileWriter(self.logs_dir, sess.graph)

            for epoch_i in range(1, self.epochs + 1):

                start_time = time.time()
                for batch_i, (input_batch, output_batch, source_sent_lengths, tar_sent_lengths) in enumerate(
                        data_utils.get_batches(x_train, y_train, self.batch_size)):

                    output_batch = np.reshape(np.array(output_batch),(-1,1))
                    #print(input_batch.shape, output_batch.shape, source_sent_lengths, tar_sent_lengths)
                    try:
                        iter_i += 1
                        #print('yo')
                        _, _summary= sess.run(
                            [self.train_op, self.summary_op],
                            feed_dict={self.input_data: input_batch,
                                       self.target_data: output_batch,
                                       self.lr: learning_rate,
                                       #self.source_sentence_length: [self.encoder_embeddings_matrix.shape[1] for x in range(self.batch_size)],
                                       self.target_sentence_length: [1 for x in range(self.batch_size)],
                                       self.keep_prob: self.dropout_keep_prob,
                                       self.lambda_coeff: lambda_val,
                                       self.z_temperature: self.z_temp,
                                       self.word_dropout_keep_prob: self.word_dropout_keep_probability,
                                       self.attention_temperature: self.attention_temp,
                                       self.gamma_coeff: self.gamma_val
                                       })
                        #print('loss',loss)
                        writer.add_summary(_summary, iter_i)

                        # KL Annealing till some iteration
                        if iter_i <= 3000:
                            lambda_val = np.round((np.tanh((iter_i - 4500) / 1000) + 1) / 2, decimals=6)

                    except Exception as e:
                        print(e)
                        # print(iter_i, e)
                        pass

                self.validate(sess, x_val, y_val, true_val)
                # BLEU = MSE
                val_bleu_str = str(self.epoch_bleu_score_val['1'][epoch_i - 1])# + ' | ' \
                               # + str(self.epoch_bleu_score_val['2'][epoch_i - 1]) + ' | ' \
                               # + str(self.epoch_bleu_score_val['3'][epoch_i - 1]) + ' | ' \
                               # + str(self.epoch_bleu_score_val['4'][epoch_i - 1])
                print('val_bleu_str',val_bleu_str)
                # Reduce learning rate, but not below its minimum value
                learning_rate = np.max([self.min_learning_rate, learning_rate * self.learning_rate_decay])

                saver = tf.train.Saver()
                saver.save(sess, self.model_checkpoint_dir + str(epoch_i) + ".ckpt")
                end_time = time.time()

                # Save the validation BLEU scores so far
                # with open(self.bleu_path + '.pkl', 'wb') as f:
                #     pickle.dump(self.epoch_bleu_score_val, f)

                self.log_str.append('Epoch {:>3}/{} - Time {:>6.1f} BLEU: {}'.format(epoch_i,
                                                                                     self.epochs,
                                                                                     end_time - start_time,
                                                                                     val_bleu_str))
                with open('logs.txt', 'w') as f:
                    f.write('\n'.join(self.log_str))
                print(self.log_str[-1])

    def validate(self, sess, x_val, y_val, true_val):
        # Calculate BLEU on validation data
        hypotheses_val = []
        references_val = []
        symbol=[]
        if self.config['experiment'] == 'qgen':
            symbol.append('?')
        
        def softmax(x):
                """Compute softmax values for each sets of scores in x."""
                return np.exp(x) / np.sum(np.exp(x), axis=0)
                
        for batch_i, (input_batch, output_batch, source_sent_lengths, tar_sent_lengths,true_val_tmp) in enumerate(
                data_utils.get_batches(x_val, y_val, self.batch_size,true_val)):
            output_batch = np.reshape(np.array(output_batch),(-1,1))
            #print('val')
            #print(input_batch.shape, output_batch.shape, source_sent_lengths, tar_sent_lengths)
            answer_logits = sess.run(self.inference_logits,
                                     feed_dict={self.input_data: input_batch,
                                                #self.target_data:output_batch, 
                                                #self.target_sentence_length: [1 for x in range(self.batch_size)],
                                                #self.source_sentence_length: [1 for x in range(self.batch_size)], #source_sent_lengths
                                                self.keep_prob: 1.0,
                                                self.word_dropout_keep_prob: 1.0,
                                                self.z_temperature: self.z_temp,
                                                self.attention_temperature: self.attention_temp})
            
            #answer_logits = softmax(answer_logits) #np.concatenate(answer_logits).astype(np.float32)
            #print('answer_logits', min(answer_logits), max(answer_logits))
            hypotheses_val.append(answer_logits)
            references_val.append(output_batch)
            # for k, pred in enumerate(answer_logits):
            #     hypotheses_val.append(
            #         word_tokenize(
            #             " ".join([self.decoder_idx_word[i] for i in pred if i not in [self.pad, -1, self.eos]])) + symbol)
            #     references_val.append([word_tokenize(true_val[batch_i * self.batch_size + k])])
        #print(max(answer_logits))
        #print(min(answer_logits))
        #bleu_scores = eval_utils.calculate_bleu_scores(references_val, hypotheses_val)
        #lin = np.linspace(0,1,11)
        references_val = np.array(references_val).reshape(1,-1)[0]
        hypotheses_val = np.array(hypotheses_val).reshape(1,-1)[0]
        true_val = np.array(true_val).reshape(1,-1)[0]
        true_val =true_val[:len(hypotheses_val)]
        # print('references_val',references_val)
        # print('hypotheses_val',hypotheses_val)
        # print('true', true_val)
        print("references_val",max(references_val),min(references_val))
        print("hypotheses_val",max(hypotheses_val),min(hypotheses_val))
        pd.DataFrame({'hyp':hypotheses_val,'ref':references_val,'real':true_val}).to_csv('../data/hypotheses_val.csv')
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(hypotheses_val)
        hypotheses_val = ecdf(hypotheses_val)#np.where(ecdf(hypotheses_val)>= 0.5,1,0)
        # ecdf_real = ECDF(true_val)
        # true_val = ecdf(true_val)
        # from collections import Counter    
        from sklearn.metrics import accuracy_score, mean_squared_error
        print('mean_squared_error', mean_squared_error(hypotheses_val,references_val))
        print('corr', np.corrcoef(references_val,hypotheses_val)[0,1])
        bleu_scores = accuracy_score(np.where(references_val>0.,1,0), np.where(hypotheses_val>0.5,1,0))
        self.epoch_bleu_score_val['1'].append(np.corrcoef(references_val,hypotheses_val)[0,1])        
        # self.epoch_bleu_score_val['2'].append(bleu_scores[1])
        # self.epoch_bleu_score_val['3'].append(bleu_scores[2])
        # self.epoch_bleu_score_val['4'].append(bleu_scores[3])




    def predict(self, checkpoint, x_test, y_test, true_val):
        pred_logits = []
        hypotheses_val = []
        references_val = []
        symbol=[]
        if self.config['experiment'] == 'qgen':
            symbol.append('?')
            
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            tf.set_random_seed(1)
            saver.restore(sess, checkpoint)

            # for batch_i, (input_batch, output_batch, source_sent_lengths, tar_sent_lengths) in enumerate(
            #         data_utils.get_batches(x_test, y_test, self.batch_size)):
            #     result = sess.run(self.inference_logits, feed_dict={self.input_data: input_batch,
            #                                                        # self.source_sentence_length: source_sent_lengths,
            #                                                         self.keep_prob: 1.0,
            #                                                         self.word_dropout_keep_prob: 1.0,
            #                                                         self.z_temperature: self.z_temp,
            #                                                         self.attention_temperature: self.attention_temp})

            #     pred_logits.extend(result)

            #     for k, pred in enumerate(result):
            #         hypotheses_test.append(
            #             word_tokenize(" ".join(
            #                 [self.decoder_idx_word[i] for i in pred if i not in [self.pad, -1, self.eos]])) + symbol)
            #         references_test.append([word_tokenize(true_test[batch_i * self.batch_size + k])])

            # bleu_scores = eval_utils.calculate_bleu_scores(references_test, hypotheses_test)
            for batch_i, (input_batch, output_batch, source_sent_lengths, tar_sent_lengths,true_val_tmp) in enumerate(
                data_utils.get_batches(x_test, y_test, self.batch_size,true_val)):
                output_batch = np.reshape(np.array(output_batch),(-1,1))
                #print('val')
                #print(input_batch.shape, output_batch.shape, source_sent_lengths, tar_sent_lengths)
                answer_logits = sess.run(self.inference_logits,
                                         feed_dict={self.input_data: input_batch,
                                                    #self.target_data:output_batch, 
                                                    #self.target_sentence_length: [1 for x in range(self.batch_size)],
                                                    #self.source_sentence_length: [1 for x in range(self.batch_size)], #source_sent_lengths
                                                    self.keep_prob: 1.0,
                                                    self.word_dropout_keep_prob: 1.0,
                                                    self.z_temperature: self.z_temp,
                                                    self.attention_temperature: self.attention_temp})
                
                #answer_logits = softmax(answer_logits) #np.concatenate(answer_logits).astype(np.float32)
                #print('answer_logits', min(answer_logits), max(answer_logits))
                hypotheses_val.append(answer_logits)
                references_val.append(output_batch)
                # for k, pred in enumerate(answer_logits):
                #     hypotheses_val.append(
                #         word_tokenize(
                #             " ".join([self.decoder_idx_word[i] for i in pred if i not in [self.pad, -1, self.eos]])) + symbol)
                #     references_val.append([word_tokenize(true_val[batch_i * self.batch_size + k])])
            #print(max(answer_logits))
            #print(min(answer_logits))
            #bleu_scores = eval_utils.calculate_bleu_scores(references_val, hypotheses_val)
            #lin = np.linspace(0,1,11)
            references_val = np.array(references_val).reshape(1,-1)[0]
            hypotheses_val = np.array(hypotheses_val).reshape(1,-1)[0]
            true_val = np.array(true_val).reshape(1,-1)[0]
            true_val =true_val[:len(hypotheses_val)]
            # print('references_val',references_val)
            # print('hypotheses_val',hypotheses_val)
            # print('true', true_val)
            print("references_val",max(references_val),min(references_val))
            print("hypotheses_val",max(hypotheses_val),min(hypotheses_val))
            pd.DataFrame({'hyp':hypotheses_val,'ref':references_val,'real':true_val}).to_csv('../data/hypotheses_val.csv')
            #from statsmodels.distributions.empirical_distribution import ECDF
            #ecdf = ECDF(hypotheses_val)
            #hypotheses_val = ecdf(hypotheses_val)#np.where(ecdf(hypotheses_val)>= 0.5,1,0)
            # ecdf_real = ECDF(true_val)
            # true_val = ecdf(true_val)
            # from collections import Counter    
            from sklearn.metrics import accuracy_score, mean_squared_error
            print('mean_squared_error', mean_squared_error(hypotheses_val,references_val))
            print('corr', np.corrcoef(references_val,hypotheses_val)[0,1])
            print(accuracy_score(np.where(references_val>0.5,1,0), np.where(hypotheses_val>0.5,1,0)))
            import matplotlib.pyplot as plt
            preds = np.where(hypotheses_val < 0.5,-1,np.where(hypotheses_val > 0.5,1,0))
            #preds = np.where(hypotheses_val>0.5,1,-1)
            plt.plot(np.cumsum(preds[preds != 0] * references_val[preds != 0]))
            plt.show() 
            #self.epoch_bleu_score_val['1'].append(bleu_scores)        
        # self.epoch_bleu_score_val['2'].append(bleu_scores[1])
        # self.epoch_bleu_score_val['3'].append(bleu_scores[2])
        # self.epoch_bleu_score_val['4'].append(bleu_scores[3])

        #print('BLEU 1 to 4 : {}'.format(' | '.join(map(str, bleu_scores))))

        return hypotheses_val

    def show_output_sentences(self, preds, y_test, input_test, true_test):
        symbol=[]
        if self.config['experiment'] == 'qgen':
            symbol.append('?')
        for k, (pred, actual) in enumerate(zip(preds, y_test)):
            print('Input:      {}'.format(input_test[k].strip()))
            print('Actual:     {}'.format(true_test[k].strip()))
            print('Generated: {}\n'.format(
                " ".join([self.decoder_idx_word[i] for i in pred if i not in [self.pad, self.eos]] + symbol)))

    def get_diversity_metrics(self, checkpoint, x_test, y_test, num_samples=10, num_iterations = 3):

        x_test_repeated = np.repeat(x_test, num_samples, axis=0)
        y_test_repeated = np.repeat(y_test, num_samples, axis=0)

        entropy_list =[]
        uni_diversity = []
        bi_diversity = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            for _ in tqdm(range(num_iterations)):
                total_ent = 0
                uni = 0
                bi = 0
                answer_logits = []
                pred_sentences = []

                for batch_i, (input_batch, output_batch, source_sent_lengths, tar_sent_lengths) in enumerate(
                        data_utils.get_batches(x_test_repeated, y_test_repeated, self.batch_size)):
                    result = sess.run(self.inference_logits, feed_dict={self.input_data: input_batch,
                                                                    self.source_sentence_length: source_sent_lengths,
                                                                    self.keep_prob: 1.0,
                                                                    self.word_dropout_keep_prob: 1.0,
                                                                    self.z_temperature: self.z_temp,
                                                                    self.attention_temperature: self.attention_temp})
                    answer_logits.extend(result)

                for idx, (actual, pred) in enumerate(zip(y_test_repeated, answer_logits)):
                    pred_sentences.append(" ".join([self.decoder_idx_word[i] for i in pred if i != self.pad][:-1]))

                    if (idx + 1) % num_samples == 0:
                        word_list = [word_tokenize(p) for p in pred_sentences]
                        corpus = [item for sublist in word_list for item in sublist]
                        total_ent += eval_utils.calculate_entropy(corpus)
                        diversity_result = eval_utils.calculate_ngram_diversity(corpus)
                        uni += diversity_result[0]
                        bi += diversity_result[1]

                        pred_sentences = []

                entropy_list.append(total_ent / len(x_test))
                uni_diversity.append(uni / len(x_test))
                bi_diversity.append(bi / len(x_test))

        print('Entropy = {:>.3f} | Distinct-1 = {:>.3f} | Distinct-2 = {:>.3f}'.format(np.mean(entropy_list),
                                                                                       np.mean(uni_diversity),
                                                                                       np.mean(bi_diversity)))
