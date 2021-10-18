import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm
from nltk import word_tokenize
from scipy.spatial.distance import cosine,euclidean, cityblock

model = gensim.models.KeyedVectors.load_word2vec_format('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/GoogleNews-vectors-negative300.bin', binary=True)

def sent2vec(s):
        M = []
        for w in s.split():
            try:
                M.append(model[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())
    
class differenced_sent:

    def __init__(self, my_dataframe):
        self.my_dataframe = my_dataframe

    def find_pre_abs(self,s1,s2):
        s1 = set(s1.split())
        s2 = set(s2.split())
        pre_abs = len(s1.difference(s2)) + len(s2.difference(s1)) 
        return pre_abs

    def find_pre_pre(self,s1,s2):
        s1 = set(s1.split())
        s2 = set(s2.split())
        pre_pre = len(s1 & s2)
        return pre_pre


    def find_question(self,s1,s2):
        s1 = set(s1.split())
        s2 = set(s2.split())    
        s1 = ' '.join(list(s1.difference(s2))) 
        return s1    

    def create_features(self):
        self.my_dataframe['pre_pre'] = self.my_dataframe.apply(lambda x: self.find_pre_pre(str(x.question1),str(x.question2)), axis=1)
        self.my_dataframe['pre_abs'] = self.my_dataframe.apply(lambda x: self.find_pre_abs(str(x.question1),str(x.question2)), axis=1)
        self.my_dataframe['question_1'] = self.my_dataframe.apply(lambda x: self.find_question(str(x.question1),str(x.question2)), axis=1)
        self.my_dataframe['question_2'] = self.my_dataframe.apply(lambda x: self.find_question(str(x.question2),str(x.question1)), axis=1)

        self.my_dataframe = self.my_dataframe[['id','question_1','question_2','pre_pre','pre_abs','is_duplicate']]


    
class mech_measures:
    
    def __init__(self, my_dataframe):
        self.my_dataframe = my_dataframe
        self.question1_vectors = np.zeros((self.my_dataframe.shape[0], 300))
        self.question2_vectors = np.zeros((self.my_dataframe.shape[0], 300))
        
    
    def convert_sentence_to_vector(self):
    
        for i, q in tqdm(enumerate(self.my_dataframe.question_1.values)):
            self.question1_vectors[i, :] = sent2vec(str(q))

        for i, q in tqdm(enumerate(self.my_dataframe.question_2.values)):
            self.question2_vectors[i, :] = sent2vec(str(q))
        

    def euclid_distance(self):
        self.my_dataframe['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(self.question1_vectors),
                                                                       np.nan_to_num(self.question2_vectors))]

if __name__ == '__main__':
    
    #df = pd.read_csv("/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/processed/Cleaned_text.csv")
    #print('Initialising differenced_set')
    #ds = differenced_sent(df)
    #print('Reducing Sentences')
    #ds.create_features()

    #df = ds.my_dataframe
    #df.to_csv('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/processed/reduced_data.csv',index = False)
    #print(df.head())

    #Assign the the reduced_data here
    df = pd.read_csv('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/processed/reduced_data.csv')
    df = df.fillna('Undefined')

    print('Initialising Mech_Measures')
    mm = mech_measures(df)
    print("Generating Vectors")
    mm.convert_sentence_to_vector()
    print("Finding Distance")
    mm.euclid_distance()
    #print(mm.question1_vectors)

    # Assign the address here
    a.to_csv('/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/processed/dissimilarity.csv',index = False)    
    
     
    #print(a.head())                                                              

