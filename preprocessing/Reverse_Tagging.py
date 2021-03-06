import codecs
import os
import re
from bs4 import BeautifulSoup
from htmltable import Table_Info


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
    def remove_space_ugly(str_line):
        # still some space can not be removed, e.g.:
        # 情况：         2018年
        # 股本的6.00%        。
        # it's ugly, but as a temporary treatment
        # TODO: fix this later. it's low priority
        tmp_str = str_line.replace(' ', '')
        return tmp_str

    @staticmethod
    def remove_space_between_chinese_character(str_line):
        patten = re.compile(r'([\w\u4e00-\u9fa5]{1})\s+([\u4e00-\u9fa5()/]{1})')
        tmp_str1 = patten.sub(r'\1\2', str_line).strip()
        # print('step 1:::' + tmp_str1)
        patten = re.compile(r'([\u4e00-\u9fa5]{1})\s+([\u4e00-\u9fa5\w]{1})\s+')
        tmp_str1 = patten.sub(r'\1\2', tmp_str1).strip()
        # print('step 2:::' + tmp_str1)
        patten = re.compile(r'([\u4e00-\u9fa5]{1})\s+([\d]{1})')
        tmp_str1 = patten.sub(r'\1\2', tmp_str1).strip()
        # print('step 3:::' + tmp_str1)
        return tmp_str1

    @staticmethod
    def normalize_punctuations(str_line):
        tmp_str = str_q2b(str_line)
        # if remove_space:
        #     tmp_str = tmp_str.replace(' ', '')
        return tmp_str

    @staticmethod
    def extract_content_from_html(str_html_line):
        # from training perspective, html tags should not be included, only the content
        # itself should be the input. all the text content are within <div type="content">...</div>
        content_str = ''
        soup = BeautifulSoup(str_html_line, "html.parser")
        # find <div type="content">
        contents = soup.find_all("div", {"type": "content"})
        for content in contents:
            content_children = content.findChildren(recursive=False)
            for child in content_children:
                # what I can think of right now is the image tag. so <img> can just be removed.
                if child.name.lower() == 'img'\
                        or child.name.lower() == 'hidden' \
                        or child.name.lower() == 'div' \
                        or child.name.lower() == 'br' \
                        or child.name.lower() == 'table':
                    # div should be an html error, because <div type='content'> should be the lowest level
                    child.decompose()
                elif child.name.lower() == 'i'\
                        or child.name.lower() == 'u':
                    content_str += child.text.strip() + ','
                else:
                    # maybe delete all sub tags in <div type="content"> ???
                    raise Exception(child.name.lower()+' is not expected in <div type="content">')
            content_str += content.text.strip()
        return content_str

    def process_common(self):
        # tmp_line = self.remove_space_between_chinese_character(self.raw_html_content)
        tmp_line = self.remove_space_ugly(self.raw_html_content)
        tmp_line = self.normalize_punctuations(tmp_line)
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
    def __init__(self, pair_list, content_string, tag_list):
        self.pair_list = pair_list
        self.content_string = content_string
        self.tag_list = tag_list

    @classmethod
    def load_from_file(cls, tag_file_path):
        pair_list = list()
        content_string = list()
        tag_list = list()
        with codecs.open(tag_file_path, 'r', 'utf-8') as f:
            for line in f:
                line = line.rstrip()
                tmp_pair = line.split('\t')
                pair_list.append(tmp_pair)
                content_string.append(tmp_pair[0])
                tag_list.append(tmp_pair[1])
        return cls(pair_list=pair_list, content_string=''.join(content_string), tag_list=tag_list)

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
        return cls(pair_list=pair_list, content_string=html_str, tag_list=tag_list)

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
        if tag_type == 'SHS':
            # long name: 重庆小康控股有限公司, short name: 小康控股. so the short name may
            # partially overwrite the long name
            match = re.finditer(escape_reg_string(training_result_str), self.content_string)
            for m in match:
                index_start = m.start()
                index_end = m.end()
                self.write_to_tag_list(index_start, index_end, tag_type, over_write=False)
                has_match = True

        if not has_match and tag_type in ['PRC', 'AMT', 'CHT']:
            # the value in html usually has been converted.
            str_another_format = "{:.4f}".format(float(training_result_str))
            match = re.finditer(str_another_format, self.content_string)
            for m in match:
                index_start = m.start()
                index_end = m.end()
                self.write_to_tag_list(index_start, index_end, tag_type)
                has_match = True

        if not has_match:
            # if the .0000 format doesn't match then, try original
            match = re.finditer(escape_reg_string(training_result_str), self.content_string)
            for m in match:
                index_start = m.start()
                index_end = m.end()
                self.write_to_tag_list(index_start, index_end, tag_type)
                has_match = True
        if not has_match:
            raise Exception(training_result_str + ' --- does not match anything')

    def write_to_tag_list(self, start_pos, end_pos, tag_type, over_write=True):
        has_conflict = False
        if not over_write:
            for i in range(end_pos-start_pos):
                if i == 0:
                    if self.tag_list[start_pos] != 'O' and \
                            self.tag_list[start_pos] != 'B-' + tag_type:
                        has_conflict = True
                else:
                    if self.tag_list[start_pos] != 'O' and \
                            self.tag_list[start_pos+i] != 'I-' + tag_type:
                        has_conflict = True
        # tag type is the tag without prefix e.g. SHL (B-SHL)
        if over_write or not has_conflict:
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

            out_put_dest_with_table = os.path.join(output_path, 'with_table', '.'.join([train_ref_results['id'], 'html']))
            if os.path.exists(out_put_dest_with_table):
                # this html contains tables and no need for tagging
                count += 1
                continue

            # load raw html file
            raw_html_file_path = os.path.join(training_data_source_path, '.'.join([train_ref_results['id'], 'html']))
            raw_html_str = get_file_content_as_string(raw_html_file_path)

            # separate the HTMLs into 2 group: with table and without table.
            # for HTMLs without table, extract the content from html. the out put should not contains any html tags
            # for HTMLs with VALID table, suppose all the required data can be found in the table!!! or, the training
            # logic can be: extract the content part and try to tag, but leave the table unchanged, since the table
            # will be html. only the data in the td cells should be cleaned.

            # normalized html content.
            # normalized_content_str = pre_processor.process_common()
            # table_processor = Table_Info.TableInfoExtractor(normalized_content_str, Table_Info.model)
            # for testing purpose only: to many errors in the html tables. skip the table processing first.
            # if table_processor.has_dataframe():
            # # if table_processor.has_valid_form():
            #     # save the preprocessed html, just leave the html to regex
            #     with codecs.open(out_put_dest_with_table, mode='w', encoding='utf-8') as f1:
            #         f1.write(normalized_content_str)
            # else:
                # load tagged file if available. Because, a doc id may have multiple records line.
                # each record line will start a process to tag the content, which means the tagged
                # file will be updated multiple times.
            tag_file_path = os.path.join(output_path, 'content_without_html_tag', '.'.join([train_ref_results['id'], 'tag']))
            try:
                tagged_content_pair = ContentTagPair.load_from_file(tag_file_path)
            except FileNotFoundError:
                # it's the first time to tag this content, so create
                pure_content = PreProcessor.extract_content_from_html(raw_html_str)
                pre_processor = PreProcessor(pure_content)
                pure_content = pre_processor.process_common()
                tagged_content_pair = ContentTagPair.init_from_string(pure_content)

            for key in keys:
                if key == 'id':
                    continue
                y_train = train_ref_results.get(key)
                if y_train == '0':
                    # some field will contains '0' as the value which doestn't make any sense.
                    y_train = None
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


raw_tag_source_folder = 'C:\\project\\AI\\project_info_extract\\data\\output\\content_without_html_tag'
# raw_tag_source_folder = 'C:\\project\\AI\\project_info_extract\\data\\output\\test_content'
training_set_output_folder = 'C:\\project\\AI\\project_info_extract\\data\\output\\training_set_sparse'
# training_set_output_folder = 'C:\\project\\AI\\project_info_extract\\data\\output\\test_dest'


def generate_training_set_dense(source_path, dest_path):
    for tag_file in os.listdir(source_path):
        print('processing ' + tag_file + '...')
        in_file_path = os.path.join(source_path, tag_file)
        out_file_path = os.path.join(dest_path, tag_file)
        tagged_content_pair = ContentTagPair.load_from_file(in_file_path)
        sentence_list = tagged_content_pair.content_string.split('。')  # separate by chinese full stop sign
        start_pos = 0

        for line in sentence_list:
            has_valid_tag = False
            # Important: 在标注的时候注意实体间的关系，主键需要在统一句话中才标注，
            # 其他属性与部分主键同时出现才标注，这样可以控制标注数据集的假阳性。
            line_length = len(line)
            if line_length == 0:
                continue
            end_pos = start_pos + line_length + 1 # to include '。'
            tag_line = tagged_content_pair.tag_list[start_pos:end_pos]
            # 股东全称       B-SHL I-SHL   ---key
            # 股东简称       B-SHS I-SHS   ---key 股东全称, 股东简称 二选一
            # 变动截止日期    B-CHD I-CHD   ---key
            # 变动价格       B-PRC I-PRC
            # 变动数量       B-AMT I-AMT
            # 变动后持股数    B-CHT I-CHT
            # 变动后持股比例   B-CPS I-CPS
            if 'B-SHL' in tag_line or 'B-SHS' in tag_line:
                if 'B-CHD' in tag_line:
                    # scenario 1: this line contains all the keys, and it should be included in the training set.
                    # because '。' doesn't take into account, so the end_pos should increase by 1.
                    has_valid_tag = True
            elif 'B-SHL' in tag_line\
                    or 'B-SHS' in tag_line\
                    or 'B-CHD' in tag_line:
                if 'B-PRC' in tag_line \
                        or 'B-AMT' in tag_line \
                        or 'B-CHT' in tag_line \
                        or 'B-CPS' in tag_line:
                    # scenario 2: this line part of the keys as well as other attributes
                    has_valid_tag = True
            if has_valid_tag:
                with codecs.open(out_file_path, mode='a+', encoding='utf-8') as f:
                    pair_list = tagged_content_pair.pair_list[start_pos:end_pos]
                    for pair in pair_list:
                        tmp_str = '\t'.join(pair)
                        print(tmp_str, file=f)
            start_pos = end_pos


def generate_training_set_sparse(source_path, dest_path):
    # to keep all 'O' sentences
    for tag_file in os.listdir(source_path):
        print('processing ' + tag_file + '...')
        in_file_path = os.path.join(source_path, tag_file)
        out_file_path = os.path.join(dest_path, tag_file)
        tagged_content_pair = ContentTagPair.load_from_file(in_file_path)
        sentence_list = tagged_content_pair.content_string.split('。')  # separate by chinese full stop sign
        start_pos = 0
        with codecs.open(out_file_path, mode='a+', encoding='utf-8') as f:
            for line in sentence_list:
                has_valid_tag = False
                # Important: 在标注的时候注意实体间的关系，主键需要在统一句话中才标注，
                # 其他属性与部分主键同时出现才标注，这样可以控制标注数据集的假阳性。
                line_length = len(line)
                if line_length == 0:
                    continue
                end_pos = start_pos + line_length + 1 # to include '。'
                tag_line = tagged_content_pair.tag_list[start_pos:end_pos]
                # 股东全称       B-SHL I-SHL   ---key
                # 股东简称       B-SHS I-SHS   ---key 股东全称, 股东简称 二选一
                # 变动截止日期    B-CHD I-CHD   ---key
                # 变动价格       B-PRC I-PRC
                # 变动数量       B-AMT I-AMT
                # 变动后持股数    B-CHT I-CHT
                # 变动后持股比例   B-CPS I-CPS
                if 'B-SHL' in tag_line or 'B-SHS' in tag_line:
                    if 'B-CHD' in tag_line:
                        # scenario 1: this line contains all the keys, and it should be included in the training set.
                        # because '。' doesn't take into account, so the end_pos should increase by 1.
                        has_valid_tag = True
                elif 'B-SHL' in tag_line\
                        or 'B-SHS' in tag_line\
                        or 'B-CHD' in tag_line:
                    if 'B-PRC' in tag_line \
                            or 'B-AMT' in tag_line \
                            or 'B-CHT' in tag_line \
                            or 'B-CPS' in tag_line:
                        # scenario 2: this line part of the keys as well as other attributes
                        has_valid_tag = True

                pair_list = tagged_content_pair.pair_list[start_pos:end_pos]
                for pair in pair_list:
                    if not has_valid_tag:
                        pair[1] = 'O'   # reset to O
                    tmp_str = '\t'.join(pair)
                    print(tmp_str, file=f)

                start_pos = end_pos

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
# training_reference_results_file = 'C:\\project\\AI\\project_info_extract\\data\\output\\test_source\\bug.txt'
training_reference_results_file = 'C:\\project\\AI\\project_info_extract\\data\\' \
                                  '[new] FDDC_announcements_round1_train_result_20180616\\zengjianchi.train'

tag_output_path = 'C:\\project\\AI\\project_info_extract\\data\\output'
process_log_file = 'C:\\project\\AI\\project_info_extract\\data\\log\\process.log'
error_log_file = 'C:\\project\\AI\\project_info_extract\\data\\log\\error.log'


if __name__ == '__main__':
    # file_path = 'C:\\project\\AI\\project_info_extract\\data\\' \
    #             'FDDC_announcements_round1_train_data\\增减持\\html\\20596892.html'
    # raw_html_str = get_file_content_as_string(file_path)
    # proc = PreProcessor(raw_html_str)
    # tmp_str = proc.process_common()
    # tmp_str = proc.remove_html_tags(tmp_str)
    # print(tmp_str)

    # process_reverse_tagging(training_reference_results_file, data_source_zjc,
    #                         tag_output_path, process_log_file, error_log_file)

    # generate_training_set_dense(raw_tag_source_folder, training_set_output_folder)
    generate_training_set_sparse(raw_tag_source_folder, training_set_output_folder)
    #
    # original_str = 'a  a 我我我  我我  我   我   sf   ssf我我  我   我   sf我我  我   我   sf我我  我   我   sf'
    # original_str = '<tr><td>增持主体</td><td>增持时间</td><td>增持方式</td><td>增持股数             (股)</td><td>增持均价(元             /股)'
    # original_str += '自 2018 年 2 月 2 日起持续通过上'
    # original_str = '减持计划的实施结果情况：         2018        年         5        月         3        日公司收到控股股东依顿投' \
    #                '资出具的《关于股份减持情况告知函》，其本次减持股份计划已实施完毕。在减持计划实施期间内，依顿投资通过上海证券交易所大宗交易、' \
    #                '集中竞价方式累计减持本公司股份         59,857,872        股，占公司总股本的         6.00%        。' \
    #                '截止         2018        年         5        月         3        日，依顿投资持有公司股份         ' \
    #                '722,182,128        股，占公司总股本的         72.38%        。截至本公告日，本次减持计划披露的减持期间届满，' \
    #                '本次减持计划已实施完毕，本次减持符合承'
    #
    # pp = PreProcessor(original_str)
    # # tmp_str = PreProcessor.remove_space_between_chinese_character(original_str)
    # print(pp.remove_space_between_chinese_character(original_str))
