import codecs
import os


def str_q2b(ustring):
    """全角转半角"""
    rs_tring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:                            # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:                 # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rs_tring += chr(inside_code)
    return rs_tring


def get_content_list_from_file(file_path):
    # file_path = os.path.join(data_home_path, doc_id+'.html')
    with codecs.open(file_path, 'r', 'utf-8') as f:
        content = f.readlines()
    # remove \n
    return [x.strip().rsplit() for x in content]


class PreProcessor:
    def __init__(self, content):
        # self.doc_id = doc_id
        # self.data_home_path = data_home_path
        # content is a list. each element in the list maps to a line of the html file
        self.content = content

    def normalize_numbers(self, str_line):
        #
        return ''

    def normalize_dates(self, str_line):
        return ''

    def normalize_punctuations(self, str_line):
        return str_q2b(str_line)

    def pre_process(self):
        converted_content = []
        for line in self.content:
            tmp_line = None
            tmp_line = self.normalize_punctuations(line)
            tmp_line = self.normalize_dates(tmp_line)
            tmp_line = self.normalize_numbers(tmp_line)
            converted_content.append(tmp_line)
        return converted_content


data_source_zjc = 'C:\\project\\AI\\project_info_extract\\data\\FDDC_announcements_round1_train_data\\增减持\\html'


def batch_pre_process(data_home_path):
    for id_html in os.listdir(data_home_path):
        file_path = os.path.join(data_home_path, id_html)
        print(file_path)
        content = get_content_list_from_file(file_path)
        pre_processor = PreProcessor(content)
        pre_processor.pre_process()


tst_str = 'ｎｉｈａｏ，ｋｅｙｉｍａ？？？！！！'
print(str_q2b(tst_str))

# import codecs
#
# # check the word embedding file
# with codecs.open('C:\\project\\AI\\data\\cc.zh.300.vec', 'r', 'utf-8') as f:
#     count = 0
#     for i in range(1000):
#         print(f.readline())
#         print('*******************************************')
#
#
# if __name__ == '__main__':
#     pass
#
#
