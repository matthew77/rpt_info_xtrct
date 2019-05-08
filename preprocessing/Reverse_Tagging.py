import codecs
import os
import re


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


def get_file_content_as_string(file_path):
    str_content = ''
    with codecs.open(file_path, 'r', 'utf-8') as f:
        for line in f:
            str_content = str_content + line.strip().rstrip()

    return str_content.lstrip()


def get_content_list_from_file(file_path):
    # file_path = os.path.join(data_home_path, doc_id+'.html')
    with codecs.open(file_path, 'r', 'utf-8') as f:
        content = f.readlines()
    # remove \n
    return [x.strip().rstrip() for x in content]


test_str = '2011 年 1 月 1 日至 2013 年 12 月 3 1 日     建发集团未减持法拉电子股份。加计本次减持， 建发集团累计减持法拉电子 11,752,826 股，占法拉电子股份总数的 5. 22%'
test_str = test_str + ' 测试这个 11,752.5 万 股。 测试这个 10826.33万股 ｎｉｈａｏ，ｋｅｙｉｍａ？？？！！！'


def normalize_number_with_thousand_separator(num):
    return convert_number_str(num)


def normalize_number_without_thousand_separator(num):
    return convert_number_str(num, has_thousand_separator=False)


def convert_number_str(num, has_thousand_separator=True):
    pos = 1
    # e.g. 11,752,826.23万
    # str1 = 11
    str1 = num.group(pos)
    # str2 = ,752,826 千分位部分
    if has_thousand_separator:
        pos = pos + 1       # pos = 2
        str2 = num.group(pos)
        str2 = str2.replace(',', '')
    else:
        # for number without thousand separator, there will be no group.
        # pos = 1
        str2 = ''
    normal_number = int(str1+str2)

    # str3 = .23  小数部分
    pos = pos + 1
    if num.group(pos):
        str3 = num.group(pos)
        decimal_part = int(str3[1:])
        decimal_length = len(str3) - 1
        decimal_number = float(decimal_part/(10**decimal_length))
        normal_number = normal_number + decimal_number
    # str4 = 万 单位部分
    pos = pos + 1
    if num.group(pos):
        str4 = num.group(pos)
        multiple = convert_chinese_number(str4)
        normal_number = normal_number * multiple
    # return a string number with 2 decimal points
    return "{:.2f}".format(normal_number)


def convert_date_str(year_month_date):
    date_list = list()
    date_list.append(year_month_date.group(1))
    date_list.append(year_month_date.group(2).zfill(2))
    date_list.append(year_month_date.group(3).zfill(2))
    return '-'.join(date_list)


def convert_chinese_number(x):
    return {
        '万':    10**4,
        '十万':   10**5,
        '百万':   10**6,
        '千万':   10**7,
        '亿':    10**8,
        '十亿':   10**9,
    }.get(x, 1)


class PreProcessor:
    def __init__(self, content):
        # self.doc_id = doc_id
        # self.data_home_path = data_home_path
        # content is a list. each element in the list maps to a line of the html file
        self.content = content

    def normalize_numbers(self, str_line):
        # 替换千分位格式
        # (?:...) A non-capturing version of regular parentheses
        str_pattern = r'[^,\.0-9](\d{1,3})((?:,\d{3})+)(\.\d+)?(万|十万|百万|千万|亿|十亿)?'  # match 千分位
        pattern = re.compile(str_pattern)
        modified_str = pattern.sub(normalize_number_with_thousand_separator, str_line)
        # 替换 <正常数字> (万|十万|百万|千万|亿|十亿) 的情况 e.g 123456百万
        str_pattern = r'[^,\.](\d+)(\.\d+)?(万|十万|百万|千万|亿|十亿)'
        pattern = re.compile(str_pattern)
        modified_str = pattern.sub(normalize_number_without_thousand_separator, modified_str)
        return modified_str

    def normalize_dates(self, str_line):
        # Noted: the pattern yyyymmdd is not taken into account
        str_pattern = r'(\d{4})[\.\-/年\s](\d{1,2})[\.\-/月\s](\d{1,2})[\D]'
        pattern = re.compile(str_pattern)
        modified_str = pattern.sub(convert_date_str, str_line)
        return modified_str

    def normalize_punctuations(self, str_line, remove_space=True):
        tmp_str = str_q2b(str_line)
        if remove_space:
            tmp_str = tmp_str.replace(' ', '')
        return tmp_str

    def pre_process(self):
        tmp_line = self.normalize_punctuations(self.content)
        tmp_line = self.normalize_dates(tmp_line)
        tmp_line = self.normalize_numbers(tmp_line)
        return tmp_line


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
