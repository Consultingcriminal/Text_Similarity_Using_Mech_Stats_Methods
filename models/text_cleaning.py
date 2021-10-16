import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
tokenizer = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')

class text_processing:
    
    def __init__(self, my_dataframe):
        '''
         my_dataframe : pandas dataframe, all changes made in this dataframe will 
                        reflect back in original one (changes are in_place). 
        '''

        self.my_dataframe = my_dataframe
        
    # Removes puncutations and tokenize the sentence.
    def tokenize_helper(self, words):
        words = str(words).lower()
        words = tokenizer.tokenize(words)
        return words
    
    def tokenize(self):
        self.my_dataframe['question1'] = self.my_dataframe['question1'].apply(self.tokenize_helper)
        self.my_dataframe['question2'] = self.my_dataframe['question2'].apply(self.tokenize_helper)
            
            
    # Remove Stopwords 
    def stopword_helper(self, words):
        words = [w for w in words if w not in stop_words]
        return words
            
    def remove_stop_words(self):
        self.my_dataframe['question1'] = self.my_dataframe['question1'].apply(self.stopword_helper)
        self.my_dataframe['question2'] = self.my_dataframe['question2'].apply(self.stopword_helper)
        
        
    # Lemmitiazation with POS tagging
    def lemmi_helper(self, tokens):
        
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        lmtzr = WordNetLemmatizer()
        s = []
        for token, tag in pos_tag(tokens):
            lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
            s.append(lemma)
            
        return ' '.join(s)
    
    
    def do_lemmatization(self):
        self.my_dataframe['question1'] = self.my_dataframe['question1'].apply(self.lemmi_helper)
        self.my_dataframe['question2'] = self.my_dataframe['question2'].apply(self.lemmi_helper)



if __name__ == '__main__':
    df = pd.read_csv("/home/vulcan/Documents/Niggas_TP/text_similarity/text_similarity/data/raw/train.csv")
    obj = text_processing(df)
    obj.tokenize()
    obj.remove_stop_words()
    a = obj.my_dataframe
    obj.do_lemmatization()
    #contents of df will be modified. 
