from mtsoo import *
from cea import cea
from mfea import mfea
from mfeaii import mfeaii
from scipy.io import savemat
from InputHandler import *
from NeuralNet import *

DATASET_TICTACTOE = 'dataset/tic-tac-toe/tic-tac-toe.data'
DATASET_IONOSPHERE = 'dataset/ionosphere/ionosphere.data'
DATASET_CREDITSCREENING = 'dataset/credit-screening/crx.data'
DATASET_BREASTCANCER = 'dataset/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
DATASET_NBIT_INP = 'dataset/nbit/training_input'
DATASET_NBIT_OUT = 'dataset/nbit/training_output'

def callback(res):
  pass

def prepareDataSet():
  inputHandler = InputHandler()
  X_data, Y_data = inputHandler.nbit(DATASET_NBIT_INP, DATASET_NBIT_OUT)
  m = X_data.shape[1] # number of samples
  train_ratio = 0.8

  X_train = X_data[:, :int(train_ratio * m)]
  Y_train = Y_data[:, :int(train_ratio * m)]
  X_test = X_data[:, int(train_ratio * m):]
  Y_test = Y_data[:, int(train_ratio * m):]

  numberof_input = X_train.shape[0]
  numberof_output = Y_train.shape[0]

  return X_train, Y_train, X_test, Y_test

def main():
  X_train, Y_train, X_test, Y_test = prepareDataSet()
  config = load_config()
  # functions = CI_HS().functions
  task = NeuralNet(X_train, Y_train)
  functions = task.functions
  pop_size = task.pop_size
  # pop_size = None

  for exp_id in range(config['repeat']):
    print('[+] EA - %d/%d' % (exp_id, config['repeat']))
    cea(functions, config, callback, pop_size=pop_size)
    print('[+] MFEA - %d/%d' % (exp_id, config['repeat']))
    mfea(functions, config, callback, pop_size=pop_size)
    print('[+] MFEAII - %d/%d' % (exp_id, config['repeat']))
    mfeaii(functions, config, callback, pop_size=pop_size)

if __name__ == '__main__':
  main()
