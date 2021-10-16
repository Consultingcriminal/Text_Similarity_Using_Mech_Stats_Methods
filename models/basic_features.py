import pandas as pd

class basic_features:
    
    def __init__(self, my_dataframe):
        self.my_dataframe = my_dataframe
        
    def create_basic_features(self):
        self.my_dataframe['len_q1'] = self.my_dataframe.question1.apply(lambda x: len(str(x)))
        self.my_dataframe['len_q2'] = self.my_dataframe.question2.apply(lambda x: len(str(x)))
        self.my_dataframe['diff_len'] = self.my_dataframe.len_q1 - self.my_dataframe.len_q2
        self.my_dataframe['len_word_q1'] = self.my_dataframe.question1.apply(lambda x: len(str(x).split()))
        self.my_dataframe['len_word_q2'] = self.my_dataframe.question2.apply(lambda x: len(str(x).split()))
        self.my_dataframe['common_words'] = self.my_dataframe.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
        self.my_dataframe['len_char_q1'] = self.my_dataframe.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        self.my_dataframe['len_char_q2'] = self.my_dataframe.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))



if __name__ == '__main__':
    obj = basic_features(data)
    obj.create_basic_features()


