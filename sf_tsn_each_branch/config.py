FILE_LIST = 'file_list_10fold_simple.pkl'
ANNOTATION_PATH = 'k400_all_test_5.pkl'
TEST_ANNOTATION_PATH = 'test_len.json'
DATA_PATH = '../data/SF_TSN_padded/train_val/'
DATA_PATH_2 = '../data/SF_TSN_interpolated/train_val/'
TEST_DATA_PATH = '../data/SF_TSN_padded/test/'
PRED_PATH = 'k400_pred.pkl' 
MODEL_SAVE_PATH = './models/'

DEVICE = 'cuda'

FEATURE_DIM = 6400 # 6400 for SF_TSN

FEATURE_LEN = 40 # DO NOT CHANGE
TIME_UNIT = 0.25 # DO NOT CHANGE

ENCODER_HIDDEN = 1024
DECODER_HIDDEN = 128
AUX_LOSS_COEF = 0.5

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
DROP_RATE = 0.2   

GLUE_PROB = 0.3 # Probability of glueing augmentation 
INTERPOLATION_PROB = 0.2 # Probability of data of DATA_PATH_2

THRESHOLD = 0.14 # Minimum score to be event boundary
SIGMA_LIST = [-1, 0.4] # List of sigma values of gaussian filtering in validation
TEST_THRESHOLD = 0.83
GOAL_SCORE = 0.85 # Train ends when validation score gets here

PATIENCE = 10 # Patience for early stopping

NUM_WORKERS = 8
