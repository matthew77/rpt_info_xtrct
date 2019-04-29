
class PreProcessor:
    def __init__(self, doc_id, path):
        self.doc_id = doc_id
        self.file_path = path

    def normalize_numbers(self):
        pass

    def normalize_dates(self):
        pass

    def normalize_punctuations(self):
        pass


data_source_zjc = 'C:\\project\\AI\\project_info_extract\\data\\FDDC_announcements_round1_train_data\\增减持\\html'
pre_processor = PreProcessor('20596890', data_source_zjc)

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
