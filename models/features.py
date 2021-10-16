import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine,euclidean
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
stop_words = stopwords.words('english')



class features:
    
    def __init__(self, my_dataframe, gensim_model):
        '''
        my_dataframe : pandas dataframe, all changes made in this dataframe will 
                       reflect back in original one (changes are in_place).  
                       
        gensim_model : instance of gesim model,
        model = gensim.models.KeyedVectors.load_word2vec_format('../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin', binary=True)
        
        '''
        self.my_dataframe = my_dataframe
        self.gensim_model = gensim_model
        self.question1_vectors = np.zeros((self.my_dataframe.shape[0], 300))
        self.question2_vectors = np.zeros((self.my_dataframe.shape[0], 300))
        
    def create_basic_features(self):
        self.my_dataframe['len_q1'] = self.my_dataframe.question1.apply(lambda x: len(str(x)))
        self.my_dataframe['len_q2'] = self.my_dataframe.question2.apply(lambda x: len(str(x)))
        self.my_dataframe['diff_len'] = self.my_dataframe.len_q1 - self.my_dataframe.len_q2
        self.my_dataframe['len_word_q1'] = self.my_dataframe.question1.apply(lambda x: len(str(x).split()))
        self.my_dataframe['len_word_q2'] = self.my_dataframe.question2.apply(lambda x: len(str(x).split()))
        self.my_dataframe['common_words'] = self.my_dataframe.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
        
        
     
    def create_fuzzy_features(self):
        self.my_dataframe['fuzz_ratio'] = self.my_dataframe.apply(lambda x: fuzz.ratio(str(x['question1']), str(x['question2'])), axis=1)
        self.my_dataframe['fuzz_partial_ratio'] = self.my_dataframe.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
        self.my_dataframe['fuzz_partial_token_sort_ratio'] = self.my_dataframe.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        self.my_dataframe['fuzz_token_set_ratio'] = self.my_dataframe.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        self.my_dataframe['fuzz_token_sort_ratio'] = self.my_dataframe.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        
        
        
    def sent2vec(self,s, gensim_model):
        '''
        s : sentence
        gensim_model : instance of gensim model
        '''
        M = []
        for w in s.split():
            try:
                M.append(gensim_model[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())
    
    
    # Function to calculate jaccard distance
    def jaccard_similarity(self,list1, list2):
        list1 = list1.split()
        list2 = list2.split()
        s1 = set(list1)
        s2 = set(list2)
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))

 
    # Function to convert sentence to vector
    def convert_sentence_to_vector(self):
        for i, q in tqdm(enumerate(self.my_dataframe.question1.values)):
            self.question1_vectors[i, :] = self.sent2vec(str(q), self.gensim_model)
            
        for i, q in tqdm(enumerate(self.my_dataframe.question2.values)):
            self.question2_vectors[i, :] = self.sent2vec(str(q), self.gensim_model)
            
            
    def create_distance_based_features(self):
        
        self.my_dataframe['cosine_similarity'] = [1 - cosine(x, y) for (x, y) in zip(np.nan_to_num(self.question1_vectors), -  -
                                                           np.nan_to_num(self.question2_vectors))]

        self.my_dataframe['jaccard_similarity'] = [self.jaccard_similarity(str(x), str(y)) for x, y in zip(data.question1.values,data.question2.values)]

        self.my_dataframe['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(self.question1_vectors),
                                                                   np.nan_to_num(self.question2_vectors))]


if __name__ == '__main__':
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    data = pd.read_csv('Cleaned_text.csv')
    data = data.drop(['id', 'qid1', 'qid2'], axis=1)

    obj = features(data, model)  ->> Pass dataframe and model as argumnet to class features.
    obj.create_basic_features()
    obj.create_fuzzy_features()
    obj.convert_sentence_to_vector()
    obj.create_distance_based_features()


