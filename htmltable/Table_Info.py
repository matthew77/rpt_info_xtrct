from bs4 import BeautifulSoup
import codecs
import gensim
import numpy as np
from scipy.linalg import norm
import pandas as pd

soup = BeautifulSoup("<html><body><p>data</p></body></html>")


class TableInfoExtractor:
    def __init__(self, html_str):
        """
        公告类型: 股东增减持
        主键: 1-2-4
        第1列: 公告id
        第2列: 股东全称
        第3列: 股东简称
        第4列: 变动截止日期
        第5列: 变动价格
        第6列: 变动数量
        第7列: 变动后持股数
        第8列: 变动后持股比例
        """
        self.html_str = html_str
        self.dataframes = list()
        self.parse()

    def parse(self):
        soup_obj = BeautifulSoup(self.html_str, "html.parser")
        for table in soup_obj.find_all('table'):
            data_frame = self.get_dataframe_from_html_table(table)
            self.dataframes.append(data_frame)

    def is_a_valid_form(self, df):
        # check the header of the dataframe to see if it contains at least 3??? valid columns
        pass

    def has_valid_form(self):
        flag = False
        for df in self.dataframes:
            if self.is_a_valid_form(df):
                flag = True
                break
        return flag

    def get_dataframe_from_html_table(self, table_obj):
        pass


source_word_vec_path = 'C:\\project\\AI\\data\\fasttext.cc.zh.300.vec'
dest_character_vec_path = 'C:\\project\\AI\\data\\chinese_character_vec_1.txt'

model = gensim.models.KeyedVectors.load_word2vec_format(dest_character_vec_path, binary=False, encoding='utf-8')


def get_average_vec_for_term(sentence, char_model):
    vec_length = char_model.vector_size
    vec = np.zeros(vec_length)
    for char in sentence:
        vec += model[char]
    vec /= len(sentence)
    return vec


def get_term_similarity(target, base_term, char_model):
    v_target = get_average_vec_for_term(target, char_model)
    v_base_term = get_average_vec_for_term(base_term, char_model)
    cosin_distance = np.dot(v_target, v_base_term)/(norm(v_target)*norm(v_base_term))
    return cosin_distance


def get_character_vector(source_path, dest_path):
    """
    from the Chinese word vector file, to get the character vector.

    """
    with codecs.open(source_path, 'r', 'utf-8') as f, codecs.open(dest_path, 'w', 'utf-8') as f2:
        count = 0
        for line in f:
            line = line.rstrip()
            tmp_vec = line.split(' ')

            key = tmp_vec[0]
            if len(key) == 1:
                print('{0} -- find chinese character {1}'.format(count+1, key))
                print(line, file=f2)
                count += 1




# get_character_vector(source_word_vec_path, dest_character_vec_path)


strings = [
    '你在干什么',
    '你在干啥子',
    '你在做什么',
    '你好啊',
    '我喜欢吃香蕉'
]

target = '你在干啥'

for str1 in strings:
    print(str1, get_term_similarity(str1, target, model))

# fasttext testing results:
# 你在干什么 0.9016322132672677
# 你在干啥子 0.9487399797212569
# 你在做什么 0.8284558704295613
# 你好啊 0.7596153269732152
# 我喜欢吃香蕉 0.5329585583990497
