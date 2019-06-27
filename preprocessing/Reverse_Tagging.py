import codecs
import os
import re


def str_q2b(ustring):
    """全角转半角"""
    rs_tring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:                            # 全角空格直接转换 全角句号。不转换
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


test_str = '2011 年 1 月 1 日至 2013 年 12 月 3 1 日再接一个2013 年 12 月 3     建发集团未减持法拉电子股份。加计本次减持，' \
           ' 建发集团累计减持法拉电子 11,752,826 股再接一个11,752,826，占法拉电子股份总数的 5. 22%还有33.33%'
test_str = test_str + ' 测试这个 11,752.5 万 股。 测试这个 10826.33万股 ｎｉｈａｏ，ｋｅｙｉｍａ？？？！！！'

def escape_reg_string(str):
    tmp_str = str.replace('(', '\(')
    tmp_str = tmp_str.replace(')', '\)')
    return tmp_str


def normalize_number_with_thousand_separator(num):
    return convert_number_str(num)


def normalize_number_without_thousand_separator(num):
    return convert_number_str(num, has_thousand_separator=False)


def convert_percent_to_float_str(per_num):
    tmp_str = per_num.group(2)
    tmp_float = float(tmp_str)/100
    return "{}{:.4f}".format(per_num.group(1), tmp_float)


def convert_number_str(num, has_thousand_separator=True):
    pos = 2  # skip the first group, because it's a start sign.
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
    return "{}{:.4f}".format(num.group(1), normal_number)


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
        '十亿':   10**9
    }.get(x, 1)

def get_tag_type(x):
    # 股东全称       B-SHL I-SHL    tag type = SHL
    # 股东简称       B-SHS I-SHS
    # 变动截止日期    B-CHD I-CHD
    # 变动价格       B-PRC I-PRC
    # 变动数量       B-AMT I-AMT
    # 变动后持股数    B-CHT I-CHT
    # 变动后持股比例   B-CPS I-CPS
    return {
        'share_holder_full_name':       'SHL',
        'share_holder_short_name':      'SHS',
        'date':                         'CHD',
        'price':                        'PRC',
        'shares_changed':               'AMT',
        'total_shares_after_change':   'CHT',
        'percent_after_change':         'CPS'
    }.get(x, '')


class PreProcessor:
    def __init__(self, raw_html_content):
        # self.doc_id = doc_id
        # self.data_home_path = data_home_path
        # content is a list. each element in the list maps to a line of the html file
        self.raw_html_content = raw_html_content

    @staticmethod
    def normalize_numbers(str_line):
        # 替换千分位格式
        # (?:...) A non-capturing version of regular parentheses
        str_pattern = r'([^,\.0-9])(\d{1,3})((?:,\d{3})+)(\.\d+)?(万|十万|百万|千万|亿|十亿)?'  # match 千分位
        pattern = re.compile(str_pattern)
        modified_str = pattern.sub(normalize_number_with_thousand_separator, str_line)
        # 替换 <正常数字> (万|十万|百万|千万|亿|十亿) 的情况 e.g 123456百万
        str_pattern = r'([^,\.])(\d+)(\.\d+)?(万|十万|百万|千万|亿|十亿)'
        pattern = re.compile(str_pattern)
        modified_str = pattern.sub(normalize_number_without_thousand_separator, modified_str)
        return modified_str

    @staticmethod
    def normalize_percent(str_line):
        # 将12.22% 转成0.1222
        str_pattern = r'(\D)(\d{1,2}\.\d{1,2})%'  # match 千分位
        pattern = re.compile(str_pattern)
        modified_str = pattern.sub(convert_percent_to_float_str, str_line)
        return modified_str

    @staticmethod
    def normalize_dates(str_line):
        # Noted: the pattern yyyymmdd is not taken into account
        str_pattern = r'(\d{4})[\.\-/年\s](\d{1,2})[\.\-/月\s](\d{1,2})[\D]'
        pattern = re.compile(str_pattern)
        modified_str = pattern.sub(convert_date_str, str_line)
        return modified_str

    @staticmethod
    def remove_space_between_chinese_character(str_line):
        patten = re.compile(r'([\w\u4e00-\u9fa5]{1})\s+([\u4e00-\u9fa5]{1})')
        tmp_str = patten.sub(r'\1\2', str_line).strip()
        patten = re.compile(r'([\u4e00-\u9fa5]{1})\s+([\u4e00-\u9fa5\w]{1})\s+')
        tmp_str = patten.sub(r'\1\2', tmp_str).strip()
        return tmp_str

    @staticmethod
    def normalize_punctuations(str_line):
        tmp_str = str_q2b(str_line)
        # if remove_space:
        #     tmp_str = tmp_str.replace(' ', '')
        return tmp_str

    def process(self):
        tmp_line = self.normalize_punctuations(self.raw_html_content)
        tmp_line = self.normalize_dates(tmp_line)
        tmp_line = self.normalize_numbers(tmp_line)
        tmp_line = self.normalize_percent(tmp_line)
        return tmp_line


# class ReverseTagging:
#     def __init__(self, doc_id, cleaned_content, value_to_be_tagged):
#         self.doc_id = doc_id
#         self.cleaned_content = cleaned_content
#         self.value_to_be_tagged = value_to_be_tagged
#
#     def process(self):
#         # return 2 list. the first list contains the content itself
#         # the second list contains the tag
#         pass


class ContentTagPair:
    def __init__(self, pair_list, html_string, tag_list):
        self.pair_list = pair_list
        self.html_string = html_string
        self.tag_list = tag_list

    @classmethod
    def load_from_file(cls, tag_file_path):
        pair_list = list()
        html_string = list()
        tag_list = list()
        with codecs.open(tag_file_path, 'r', 'utf-8') as f:
            for line in f:
                line = line.rstrip()
                tmp_pair = line.split('\t')
                pair_list.append(tmp_pair)
                html_string.append(tmp_pair[0])
                tag_list.append(tmp_pair[1])
        return cls(pair_list=pair_list, html_string=''.join(html_string), tag_list=tag_list)

    @classmethod
    def init_from_string(cls, html_str):
        pair_list = list()
        tag_list = list()
        for char in html_str:
            tmp_pair = list()
            tmp_pair.append(char)
            tmp_pair.append('O')  # init tag will all be set to O
            pair_list.append(tmp_pair)
            tag_list.append('O')
        return cls(pair_list=pair_list, html_string=html_str, tag_list=tag_list)

    def tag(self, training_result_str, tag_type):
        # training_result_str is from training standard results
        # tag is a type: e.g. person or number for 增减持, we have following tag type (BIO tags)
        # 股东全称       B-SHL I-SHL    tag type = SHL
        # 股东简称       B-SHS I-SHS
        # 变动截止日期    B-CHD I-CHD
        # 变动价格       B-PRC I-PRC
        # 变动数量       B-AMT I-AMT
        # 变动后持股数    B-CHT I-CHT
        # 变动后持股比例   B-CPS I-CPS

        has_match = False
        if tag_type in ['PRC', 'AMT', 'CHT']:
            # the value in html usually has been converted.
            str_another_format = "{:.4f}".format(float(training_result_str))
            match = re.finditer(str_another_format, self.html_string)
            for m in match:
                index_start = m.start()
                index_end = m.end()
                self.write_to_tag_list(index_start, index_end, tag_type)
                has_match = True
        if not has_match:
            # if the .0000 format doesn't match then, try original
            match = re.finditer(escape_reg_string(training_result_str), self.html_string)
            for m in match:
                index_start = m.start()
                index_end = m.end()
                self.write_to_tag_list(index_start, index_end, tag_type)
                has_match = True
        if not has_match:
            raise Exception(training_result_str + ' --- does not match anything')

    def write_to_tag_list(self, start_pos, end_pos, tag_type):
        # tag type is the tag without prefix e.g. SHL (B-SHL)
        for i in range(end_pos-start_pos):
            if i == 0:
                self.tag_list[start_pos] = 'B-' + tag_type
            else:
                self.tag_list[start_pos+i] = 'I-' + tag_type

    def update(self):
        # synchronize the pair_list and tag_list.
        pos = 0
        for pair in self.pair_list:
            pair[1] = self.tag_list[pos]
            pos += 1

    def save(self, tag_file_path):
        with codecs.open(tag_file_path, 'w', encoding='utf-8') as f:
            for pair in self.pair_list:
                tmp_str = '\t'.join(pair)
                print(tmp_str, file=f)


def my_log(log_file_path, msg):
    print(msg)
    if os.path.exists(log_file_path):
        with open(log_file_path, 'a') as log:
            print(msg, file=log)
    else:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'w') as log:
            print(msg, file=log)


def process_reverse_tagging(training_results_file_path, training_data_source_path,
                            output_path, proc_log_file, err_log_file):
    count = 0
    if os.path.exists(proc_log_file):
        with open(proc_log_file, 'r') as log:
            last_time_count = int(log.read())
    else:
        os.makedirs(os.path.dirname(proc_log_file), exist_ok=True)
        with open(proc_log_file, 'w') as log:
            log.write('0')
        last_time_count = 0

    keys = ('id', 'share_holder_full_name', 'share_holder_short_name',
            'date', 'price', 'shares_changed', 'total_shares_after_change', 'percent_after_change')
    with codecs.open(training_results_file_path, 'r', 'utf-8') as f:
        for line in f:  # loop through all training results.
            print('processing line #{0}.......'.format(count+1))
            if last_time_count > count:
                count += 1
                continue
            line = line.replace('\r\n', '')
            values = line.split('\t')
            train_ref_results = dict(zip(keys, values))
            # load raw html file
            raw_html_file_path = os.path.join(training_data_source_path, '.'.join([train_ref_results['id'], 'html']))
            raw_html_str = get_file_content_as_string(raw_html_file_path)
            pre_processor = PreProcessor(raw_html_str)
            # TODO: separate the HTMLs into 2 group: with table and without table.
            # TODO: for HTMLs without table, extract the content from html. the out put should not contains any html tags
            # TODO: for HTMLs with VALID table, suppose all the required data can be found in the table!!! or, the training
            # TODO: logic can be: extract the content part and try to tag, but leave the table unchanged, since the table
            # TODO: will be html. only the data in the td cells should be cleaned.

            # normalized content loaded.
            normalized_content_str = pre_processor.process()

            # load tagged file if available. Because, a doc id may have multiple records line.
            # each record line will start a process to tag the content, which means the tagged
            # file will be updated multiple times.
            tag_file_path = os.path.join(output_path, '.'.join([train_ref_results['id'], 'tag']))
            try:
                tagged_content_pair = ContentTagPair.load_from_file(tag_file_path)
            except FileNotFoundError:
                # it's the first time to tag this html, so create
                tagged_content_pair = ContentTagPair.init_from_string(normalized_content_str)

            for key in keys:
                if key == 'id':
                    continue
                y_train = train_ref_results.get(key)
                if y_train:     # not all field contains value.
                    tag_type = get_tag_type(key)
                    try:
                        tagged_content_pair.tag(y_train, tag_type)
                    except Exception:
                        error_msg = train_ref_results['id'] + '::: matching ' + key + ' = ' + y_train + ' failed'
                        my_log(err_log_file, '\t'.join([train_ref_results['id'], error_msg]))

            tagged_content_pair.update()
            # save results.
            tagged_content_pair.save(tag_file_path)

            count += 1

            with open(proc_log_file, 'w') as log:
                log.write('{0}'.format(count))

            # for testing only
            #if count > 10:
            #    break

    print('Done!')


# def batch_pre_process(data_home_path):
#     for id_html in os.listdir(data_home_path):
#         file_path = os.path.join(data_home_path, id_html)
#         print(file_path)
#         content = get_content_list_from_file(file_path)
#         pre_processor = PreProcessor(content)
#         pre_processor.pre_process()


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
data_source_zjc = 'C:\\project\\AI\\project_info_extract\\data\\FDDC_announcements_round1_train_data\\增减持\\html'
# training_reference_results_file = 'C:\\project\\AI\\project_info_extract\\data\\' \
#                                   '[new] FDDC_announcements_round1_train_result_20180616\\zengjianchi.train'

# for testing purpose only
training_reference_results_file = 'C:\\project\\AI\\project_info_extract\\data\\' \
                                  '[new] FDDC_announcements_round1_train_result_20180616\\zengjianchi.train'
tag_output_path = 'C:\\project\\AI\\project_info_extract\\data\\output'
process_log_file = 'C:\\project\\AI\\project_info_extract\\data\\log\\process.log'
error_log_file = 'C:\\project\\AI\\project_info_extract\\data\\log\\error.log'


if __name__ == '__main__':
    # process_reverse_tagging(training_reference_results_file, data_source_zjc,
    #                         tag_output_path, process_log_file, error_log_file)

    original_str = 'a  a 我我我  我我  我   我   sf   ssf我我  我   我   sf我我  我   我   sf我我  我   我   sf'
    tmp_str = PreProcessor.remove_space_between_chinese_character(original_str)
    print(tmp_str)
