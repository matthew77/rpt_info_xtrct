from bs4 import BeautifulSoup
import codecs
import gensim
import numpy as np
from scipy.linalg import norm
import pandas as pd
import jieba

soup = BeautifulSoup("<html><body><p>data</p></body></html>")


class TableInfoExtractor:
    def __init__(self, html_str, word_vec_model, similarity_threshold = 0.8):
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
        self.standard_header = ['股东全称', '股东简称', '变动截止日期', '变动价格', '变动数量', '变动后持股数', '变动后持股比例']
        self.html_str = html_str
        self.similarity_threshold = similarity_threshold
        self.word_vec_model = word_vec_model
        self.dataframes = list()
        self.parse()

    def parse(self):
        soup_obj = BeautifulSoup(self.html_str, "html.parser")
        for table in soup_obj.find_all('table'):
            data_frame = self.get_dataframe_from_html_table(table)
            self.dataframes.append(data_frame)

    def is_a_valid_form(self, df, is_character_based=True):
        # check the header of the dataframe to see if it contains at least 3??? valid columns
        # is_character_based=True -- break each of the header term based on character or word(using jieba)
        matched_headers = dict()
        header = list(df.columns.values)
        for column_name in header:
            similarity_score = 0
            standard_column_name = None
            for standard in self.standard_header:
                tmp_similar_score = get_term_similarity(column_name, standard, self.word_vec_model, is_character_based)
                if tmp_similar_score > similarity_score:
                    similarity_score = tmp_similar_score
                    standard_column_name = standard
            if similarity_score > 0.8:      # 0.8 is a threshold for similarity
                matched_headers[standard_column_name] = column_name
        if len(matched_headers) >= 3:
            return True, matched_headers
        return False, matched_headers


    def has_valid_form(self):
        flag = False
        for df in self.dataframes:
            if self.is_a_valid_form(df):
                flag = True
                break
        return flag

    def get_dataframe_from_html_table(self, table_obj):
        # table_obj is a soup table object
        multi_row_header = False
        header = list()
        row_number = 1
        num_of_rows_in_head = 1
        position_stack = dict() # key=column number, value is a 2 value list [rowspan, colspan]
        for tr in table_obj.find_all('tr'):
            col_number = 1
            tds = iter(tr.find_all('td'))
            if row_number == 1:
                for td in tds:
                    # for td in tr.find_all('td'):
                    row_span = 1
                    col_span = 1
                    # head part
                    if row_number == 1 : # or row_number <= num_of_rows_in_head
                        if td.get('rowspan'):
                            # for title process, I think rowspan = 2 should be maximum
                            row_span = int(td.get('rowspan'))
                        if td.get('colspan'):
                            row_span = int(td.get('colspan'))
                        # prcessing the head info
                        td_stack = [1, 1]  # default value
                        has_span = False
                        if row_span > 1:
                            # header contains multiple lines
                            num_of_rows_in_head = row_span
                        if col_span > 1:
                            for i in range(col_span):
                                header.append(td.text)
                            row_span -= 1  # the current row is processed
                        else:
                            header.append(td.text)
                        td_stack[0] = row_span - 1  # the current row has been processed. e.g. encounter a td with
                                # rowspan=2, so after the above process, the rowspan should be 1
                        td_stack[1] = col_span
                        position_stack[col_number] = td_stack

            elif 1 < row_number <= num_of_rows_in_head:
                # the rest of the header
                for i in range(len(header)):
                    if i+1 in position_stack:
                        # the column still contains rowspan, so now you don't have relative td in this
                        # position, because it's already processed in previous row. so for this td, just
                        # keep the previous value
                        td_stack = position_stack[i+1]
                        if td_stack[0] - 1 <= 0:  # the rowspan has finished processing
                            del position_stack[i+1]
                    else:
                        # should merge the value with the previous value.
                        td = next(tds)
                        pre_value = header[i]
                        cur_value = td.text
                        header[i] = pre_value + cur_value

            else:  # process the content/value part of the table.
                pass

            row_number += 1


source_word_vec_path = 'C:\\project\\AI\\data\\fasttext.cc.zh.300.vec'
dest_character_vec_path = 'C:\\project\\AI\\data\\chinese_character_vec_1.txt'

model = gensim.models.KeyedVectors.load_word2vec_format(dest_character_vec_path, binary=False, encoding='utf-8')


def get_average_vec_for_term(sentence, char_model, is_character_based=True):
    vec_length = char_model.vector_size
    vec = np.zeros(vec_length)
    if is_character_based:
        for char in sentence:
            vec += model[char]
        vec /= len(sentence)
    else:
        # word based, use jieba to get the words
        words = jieba.lcut(sentence)
        for word in words:
            vec += model[word]
        vec /= len(words)
    return vec


def get_term_similarity(target, base_term, char_model, is_character_based=True):
    v_target = get_average_vec_for_term(target, char_model, is_character_based)
    v_base_term = get_average_vec_for_term(base_term, char_model, is_character_based)
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
    print(str1, get_term_similarity(str1, target, model, is_character_based=False))

# fasttext character based testing results:
# 你在干什么 0.9016322132672677
# 你在干啥子 0.9487399797212569
# 你在做什么 0.8284558704295613
# 你好啊 0.7596153269732152
# 我喜欢吃香蕉 0.5329585583990497

# fasttext word based testing results:
# 你在干什么 0.8147806015920263
# 你在干啥子 0.9384983249941359
# 你在做什么 0.8558632670866593
# 你好啊 0.5827293878063351
# 我喜欢吃香蕉 0.624095035853279