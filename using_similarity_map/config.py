FILE_LIST = 'file_list_10fold_main.pkl'
ANNOTATION_PATH = 'k400_all_test_5.pkl' ## contain test set boundaries predicted by our models
TEST_ANNOTATION_PATH = 'test_len.json'
DATA_PATH = '../data/SF_TSN_padded/train_val/'
DATA_PATH_2 = '../data/SF_TSN_interpolated/train_val/'
TEST_DATA_PATH = '../data/LOVEU_both_hr_padded/test/'
PRED_PATH = 'k400_pred.pkl'
MODEL_SAVE_PATH = './models/'

DEVICE = 'cuda'

FEATURE_DIM = 6400 # 6400 for SF_TSN

FEATURE_LEN = 40 # DO NOT CHANGE
TIME_UNIT = 0.25 # DO NOT CHANGE

GAP = 16 # VALID LOCAL RANGE
CHANNEL_NUM = 4 # 4 
ENCODER_HIDDEN = 512 #512
DECODER_HIDDEN = 128  #128

EVENT_LOSS_COEF = 0.5 
SHOT_LOSS_COEF = 0.2
WHOLE_LOSS_COEF = 0.3

AUX_LOSS_COEF = 0.5   

BATCH_SIZE = 192
LEARNING_RATE = 1e-4
DROP_RATE = 0.2

GLUE_PROB = 0.3 # Probability of glueing augmentation 
INTERPOLATION_PROB = 0.2 # Probability of data of DATA_PATH_2
FLIP_PROB = 0.5 # Probability of flip

THRESHOLD = 0.14 # Minimum score to be event boundary
SIGMA_LIST = [-1, 0.4] # List of sigma values of gaussian filtering in validation
TEST_THRESHOLD = 0.830
GOAL_SCORE = 0.850 # Train ends when validation score gets here

PATIENCE = 10 # Patience for early stopping

NUM_WORKERS = 8
