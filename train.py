import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

import sys

if not '../' in sys.path: sys.path.append('../')

import pandas as pd
import numpy as np
import csv 
from utils import data_utils
from model_config import config
from ved_varAttn import VarSeq2SeqVarAttnModel




def train_model(config):
    print('[INFO] Preparing data for experiment: {}'.format(config['experiment']))
    if config['experiment'] == 'qgen':
        print("next")
        train_data = np.fromfile(config['data_dir'] + 'train.csv', dtype=np.float64)
        train_data = train_data.reshape((13212,2001))
        val_data = np.fromfile(config['data_dir'] + 'val.csv', dtype=np.float64)
        val_data = val_data.reshape((6606, 2001)) #(6607, 2001)
        #train_data = pd.read_csv(config['data_dir'] + 'train.csv').fillna(0)
        #val_data = pd.read_csv(config['data_dir'] + 'val.csv').fillna(0)
        #test_data = pd.read_csv(config['data_dir'] + 'test.csv')
        true_val = val_data[:,-1]
        print("loaded")
        #true_val = np.where(val_data['forret']>0,1,0)
        #print(true_val)
        # train_data = pd.read_csv(config['data_dir'] + 'df_qgen_train.csv')
        # val_data = pd.read_csv(config['data_dir'] + 'df_qgen_val.csv')
        # test_data = pd.read_csv(config['data_dir'] + 'df_qgen_test.csv')
        # input_sentences = pd.concat([train_data['answer'], val_data['answer'], test_data['answer']])
        # output_sentences = pd.concat([train_data['question'], val_data['question'], test_data['question']])
        # true_val = val_data['question']
        # filters = '!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n'
        # w2v_path = config['w2v_dir'] + 'w2vmodel_qgen.pkl'

    elif config['experiment'] == 'dialogue':
        train_data = pd.read_csv(config['data_dir'] + 'train2.csv')
        val_data = pd.read_csv(config['data_dir'] + 'val2.csv')
        test_data = pd.read_csv(config['data_dir'] + 'test.csv')
        true_val = np.where(val_data['forret']>0,1,0)
        # train_data = pd.read_csv(config['data_dir'] + 'df_dialogue_train.csv')
        # val_data = pd.read_csv(config['data_dir'] + 'df_dialogue_val.csv')
        # test_data = pd.read_csv(config['data_dir'] + 'df_dialogue_test.csv')
        # input_sentences = pd.concat([train_data['line'], val_data['line'], test_data['line']])
        # output_sentences = pd.concat([train_data['reply'], val_data['reply'], test_data['reply']])
        # true_val = val_data['reply']
        # filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
        # w2v_path = config['w2v_dir'] + 'w2vmodel_dialogue.pkl'

    else:
        print('Invalid experiment name specified!')
        return

    print('[INFO] Tokenizing input and output sequences')
    # x, input_word_index = data_utils.tokenize_sequence(input_sentences,
    #                                                    filters,
    #                                                    config['encoder_num_tokens'],
    #                                                    config['encoder_vocab'])

    # y, output_word_index = data_utils.tokenize_sequence(output_sentences,
    #                                                     filters,
    #                                                     config['decoder_num_tokens'],
    #                                                     config['decoder_vocab'])

    # print('[INFO] Split data into train-validation-test sets')
    # x_train, y_train, x_val, y_val, x_test, y_test = data_utils.create_data_split(x,
    #                                                                               y,
    #                                                                               config['experiment'])
    from sklearn.preprocessing import MinMaxScaler
    # x_train = pd.read_csv(config['data_dir'] + 'X_train.csv',nrows=100000).fillna(0).iloc[:,1:] #
    # x_train = x_train.replace([np.inf, -np.inf], 0.)
    # scaler = MinMaxScaler().fit(np.array(x_train))
    # x_train = scaler.transform(np.array(x_train))
    # y_train = pd.read_csv(config['data_dir'] + 'y_train.csv',nrows=100000).fillna(0).iloc[:,1:]
    # y_train[y_train > 0.1] = 0.1
    # y_train[y_train < -0.1] = -0.1
    # y_train = y_train.replace([np.inf, -np.inf], 0.) #np.where(np.array(y_train.replace([np.inf, -np.inf], 0.)).reshape(-1, 1)>0,1,0)
    # scaler2 = MinMaxScaler().fit(np.array(y_train))
    # y_train = scaler2.transform(np.array(y_train))
    # x_val = pd.read_csv(config['data_dir'] + 'X_train.csv',skiprows=100000,nrows=50000).fillna(0).iloc[:,1:]
    # x_val = x_val.replace([np.inf, -np.inf], 0.)
    # x_val = scaler.transform(np.array(x_val))
    # y_val = pd.read_csv(config['data_dir'] + 'y_train.csv',skiprows=100000,nrows=50000).fillna(0).iloc[:,1:]
    # true_val = np.array(y_val.replace([np.inf, -np.inf], 0.)).reshape(-1, 1) #np.where(y_val>0,1,0)
    # true_val[true_val > 0.1] = 0.1
    # true_val[true_val < -0.1] = -0.1
    # y_val = scaler2.transform(np.array(true_val))
    #y_val =np.where(np.array(y_val.replace([np.inf, -np.inf], 0.)).reshape(-1, 1)>0,1,0)
    scaler = MinMaxScaler().fit(np.array(train_data[:,-1]).reshape(-1, 1))
    x_train, y_train, x_val, y_val =  train_data[:,:-1],scaler.transform(np.array(train_data[:,-1]).reshape(-1, 1)), \
                                      val_data[:,:-1], scaler.transform(np.array(val_data[:,-1]).reshape(-1, 1))#, test_data[:,-1], test_data['forret']
    # x_train, y_train, x_val, y_val =  train_data.iloc[:,1:-1],np.array(train_data['forret']).reshape(-1, 1), \
    #                                   val_data.iloc[:,1:-1], np.array(val_data['forret']).reshape(-1, 1)
    # # x_train, y_train, x_val, y_val =  train_data.iloc[:,1:-1],np.where(np.array(train_data['forret']).reshape(-1, 1)>0,1,0), \
    #                                   val_data.iloc[:,1:-1], np.where(np.array(val_data['forret']).reshape(-1, 1)>0,1,0)
    
    # encoder_embeddings_matrix = data_utils.create_embedding_matrix(input_word_index,
    #                                                                config['embedding_size'],
    #                                                                w2v_path)

    # decoder_embeddings_matrix = data_utils.create_embedding_matrix(output_word_index,
    #                                                                config['embedding_size'],
    #                                                                w2v_path)
    encoder_embeddings_matrix = x_train
    decoder_embeddings_matrix = y_train
    print('encoder_embeddings_matrix.shape',encoder_embeddings_matrix.shape)
    print('decoder_embeddings_matrix.shape',decoder_embeddings_matrix.shape)
    print('x_train.shape',x_train.shape)
    print('y_train.shape',y_train.shape)
    # # Re-calculate the vocab size based on the word_idx dictionary
    # config['encoder_vocab'] = len(input_word_index)
    # config['decoder_vocab'] = len(output_word_index)

    model = VarSeq2SeqVarAttnModel(config,
                                   encoder_embeddings_matrix,
                                   decoder_embeddings_matrix)#,
                                   # input_word_index,
                                   # output_word_index)

    model.train(x_train, y_train, x_val, y_val, true_val)
    # preds = model.predict(checkpoint, 
    #                   x_test, 
    #                   y_test, 
    #                   true_test, 
    #                   )

if __name__ == '__main__':
    train_model(config)
