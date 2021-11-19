from autoxgb import AutoXGB


# required parameters:
train_filename = "/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/interim/final_feature_train_1.csv"
output = "/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/models/model/output/comb_2"

# optional parameters
test_filename = None
task = None
idx = None
targets = ["is_duplicate_x"]
features = ['len_q1', 'len_q2', 'diff_len', 'len_word_q1',
       'len_word_q2', 'common_words', 'len_char_q1', 'len_char_q2',
       'fuzz_ratio', 'fuzz_partial_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_partial_token_set_ratio', 'fuzz_token_set_ratio',
       'fuzz_token_sort_ratio', 'cosine_similarity', 'jaccard_similarity',
       'euclidean_distance_x', 'wmd', 'cityblock_distance']
categorical_features = None
use_gpu = True
num_folds = 5
seed = 42
num_trials = 100
time_limit = 360
fast = False

# Now its time to train the model!
axgb = AutoXGB(
    train_filename=train_filename,
    output=output,
    test_filename=test_filename,
    task=task,
    idx=idx,
    targets=targets,
    features=features,
    categorical_features=categorical_features,
    use_gpu=use_gpu,
    num_folds=num_folds,
    seed=seed,
    num_trials=num_trials,
    time_limit=time_limit,
    fast=fast,
)
axgb.train()