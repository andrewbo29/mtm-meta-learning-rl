TASK_NAME = 'reach-v2'
ADAPT_LR = 1e-4  # alpha, inner lr
META_LR = 1e-3  # beta, outer lr
ADAPT_STEPS = 1  # number of inner gradient updates
META_BATCH_SIZE = 10  # 20 meta-batch size
ADAPT_BATCH_SIZE = 5  # 10 roll-outs per task
GAMMA = 0.99  # discount factor
MAX_KL = 1e-2  # maximum KL divergence

EPSILON = 1e-6
CRITIC_LR = 3e-4
TAU = 1.00
NUM_ITERATIONS = 1000
SEED = 42
CUDA = True

LS_LR_ALPHA = 0.5
LS_MAX_STEPS = 15
EVAL_BATCH_SIZE = 10

LOGS_ITERATION = 5
LOGS_FOLDER = 'logs/'
