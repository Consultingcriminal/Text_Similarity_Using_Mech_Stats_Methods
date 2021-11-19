import pandas as pd
import numpy as np

df = pd.read_csv('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/interim/final_feature_train.csv')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(-1, inplace=True)

df.to_csv('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/interim/final_feature_train_1.csv',index = False)

