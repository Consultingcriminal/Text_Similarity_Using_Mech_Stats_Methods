import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine,euclidean, cityblock
from nltk import word_tokenize

# change file path
model = gensim.models.KeyedVectors.load_word2vec_format('../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin', binary=True)

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
    

def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    return model.wmdistance(s1, s2)
    
    
def jaccard_similarity(list1, list2):
    list1 = list1.split()
    list2 = list2.split()
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))
    
        
class similarity_measure_features:
    
    def __init__(self, my_dataframe):
        self.my_dataframe = my_dataframe
        self.question1_vectors = np.zeros((self.my_dataframe.shape[0], 300))
        self.question2_vectors = np.zeros((self.my_dataframe.shape[0], 300))
        
    
    def convert_sentence_to_vector(self):
    
        for i, q in tqdm(enumerate(self.my_dataframe.question1.values)):
            self.question1_vectors[i, :] = sent2vec(str(q))

        for i, q in tqdm(enumerate(self.my_dataframe.question2.values)):
            self.question2_vectors[i, :] = sent2vec(str(q))
    
    
    def create_distance_based_feature(self):

            self.my_dataframe['cosine_similarity'] = [1 - cosine(x, y) for (x, y) in zip(np.nan_to_num(self.question1_vectors), -  -
                                                               np.nan_to_num(self.question2_vectors))]

            self.my_dataframe['jaccard_similarity'] = [jaccard_similarity(str(x), str(y)) for x, y in zip(self.my_dataframe.question1.values,self.my_dataframe.question2.values)]

            self.my_dataframe['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(self.question1_vectors),
                                                                       np.nan_to_num(self.question2_vectors))]
            
            self.my_dataframe['wmd'] = self.my_dataframe.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
            
            
            self.my_dataframe['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(self.question1_vectors),
                                                          np.nan_to_num(self.question2_vectors))]



if __name__ == '__main__':

   
    csv_path = ""  # Path of cleaned_text csv file
    data = pd.read_csv(csv_path)
    obj = similarity_measure_features(data)
    obj.convert_sentence_to_vector()
    obj.create_distance_based_feature()

