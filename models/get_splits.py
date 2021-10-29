import pandas as pd
import numpy as np 
from sklearn import model_selection

if __name__=="__main__":
    #df=pd.read_csv('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/interim/Final Features.csv')
    #df2 = pd.read_csv('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/processed/mechanical_features.csv')
    #df = df.merge(df2,left_on='id',right_on='id')
    #print(df.head())
    #df = df.drop('is_duplicate_y',axis = 1)
    #df.to_csv('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/interim/final_feature_train.csv',index = False)
    
    df=pd.read_csv('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/interim/final_feature_train.csv')
    df['is_duplicate_x']=df['is_duplicate_x'].astype('i8')
    df.loc[:,"kfold"]=-1
    df=df.sample(frac=1).reset_index(drop=True)

    y=df.is_duplicate_x.values
    skf=model_selection.StratifiedKFold(n_splits=5)


    for f, (t_, v_) in enumerate(skf.split(X=df,y=y)):
        df.loc[v_,"kfold"]=f

    print(df['kfold'].value_counts())    

    df.to_csv('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/external/final_features_folds.csv',index=False)    