import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class vectors:

    def __init__(self,my_dataframe):
        '''
         my_dataframe : pandas dataframe, all changes made in this dataframe will 
                        reflect back in original one (changes are in_place). 
        '''
        self.my_dataframe = my_dataframe

    def CVec_helper(self,s1,s2):
        '''
         CVec_Helper : Returns the vector of the sentence in the pair
        '''
        corpus = []
        corpus.append(s1)
        corpus.append(s2)
        vectorizer = CountVectorizer(analyzer='word')
        X = vectorizer.fit_transform(corpus)
        return (X.toarray()[0]).tolist(), (X.toarray()[1]).tolist()

    def CVec(self):
        self.my_dataframe['Vectors'] = self.my_dataframe.apply(lambda x: self.CVec_helper(x.question1, x.question2), axis=1)


class similarity:

    def __init__(self,my_dataframe,lambda_val = 0.99):
        '''
         my_dataframe : pandas dataframe, all changes made in this dataframe will 
                        reflect back in original one (changes are in_place). 
        '''
        self.my_dataframe = my_dataframe
        self.lambda_val = lambda_val

    def similarity_helper(self,list1 = [],list2 = []):
        '''
         similarity : returns the similarity score of the two sentences

        '''
        l3 = []
        pre_pre_count  = 0
        pre_abs_count = 0

        for i in range(len(list1)):
            # pre_pre pairs
            if list1[i] and list2[i] > 0: 
                l3.append(1)
                pre_pre_count += 2

            # pre_abs pairs
            elif (list1[i] > 0 and list2[i] == 0) or (list2[i] > 0 and list1[i] == 0): 
                l3.append(2)
                pre_abs_count += 1

            # pre_abs pairs
            else: 
                l3.append(0)

        if not pre_pre_count:
            return 0
        else:
            difference_score = 0
            for i in range(len(l3)):
                if l3[i] == 1: #calculating present-present pairs
                    difference_score = difference_score + 0.5*((list1[i] - list2[i])**2)
                elif l3[i] == 2: #calculating present-absent pairs
                    difference_score  = difference_score + 0.5 * ((math.pow(self.lambda_val,2))*((list1[i] - list2[i])**2))  
        
            difference = math.log(1 + (difference_score/(pre_pre_count + pre_abs_count)))
            similarity  = 1/(1 + difference)
            return similarity
    
    def find_similarity(self):
        self.my_dataframe["mech_sim_{}".format(self.lambda_val)]  = self.my_dataframe.apply(lambda x: self.similarity_helper(x.Vectors[0],x.Vectors[1]), axis=1)           

if __name__ == "__main__":
    train = pd.read_csv("/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/raw/train.csv")
    train = train.fillna('Undefined')
    vc = vectors(train)
    vc.CVec()
    a = vc.my_dataframe
    a.to_pickle("/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/processed/my_csv.pkl")

    train = pd.read_pickle("/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/processed/my_csv.pkl")
    lambda_val = 1.30
    fs = similarity(train,lambda_val)

    fs.find_similarity()
    a = fs.my_dataframe
    a.to_csv("/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/processed/my_csv_similarity_{}".format(lambda_val),index = False)
