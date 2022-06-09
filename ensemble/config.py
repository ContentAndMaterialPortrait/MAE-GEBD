# TEST_ANNOTATION_PATH = 'test_len.json' ## for base models ensemble
TEST_ANNOTATION_PATH = 'test_len_mae.json' ## for base + MAE models
# TEST_ANNOTATION_PATH = 'test_len_mae_good.json'  ## for base + MAE models, only ensemble easy samples
# TEST_ANNOTATION_PATH = 'test_len_mae_hard.json'  ## for base + MAE models, only ensemble hard samples
TEST_DATA_PATH = '../data/LOVEU_both_hr_padded/test/'
PRED_PATH = 'k400_pred.pkl'
PROB_RESULT_PATH = 'prob_results'

DEVICE = 'cuda'

FEATURE_LEN = 40
TIME_UNIT = 0.25

BATCH_SIZE = 24 #8

POOL_SIZE = 5 # 3
THRESHOLD = 0.14
