import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split
import gensim

from preprocessing.Reverse_Tagging import ContentTagPair


torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
# EMBEDDING_DIM = 5
HIDDEN_DIM = 50    # not sure how to determine this figure???


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence_with_pre_trained(seq, char_vec):
    # based on pre trained word vector such as fasttext.
    # I manually added a UNK zero vector into the fasttext vector for rare characters.
    idxs = list()
    for c in seq:
        if c in char_vec.vocab:
            idxs.append(model.vocab[c].index)
        else:
            # for all of the rare characters, replace with 'UNK'
            idxs.append(model.vocab['UNK'].index)
    return torch.tensor(idxs, dtype=torch.long)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, hidden_dim, character_vector):
        # deleted parameters: vocab_size, embedding_dim
        super(BiLSTM_CRF, self).__init__()
        # self.embedding_dim = embedding_dim
        self.embedding_dim = 300  # hard coded, since using the pre-trained word vectors.
        self.hidden_dim = hidden_dim
        # self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        pre_trained_word_vectors = torch.FloatTensor(character_vector.vectors)
        self.word_embeds = nn.Embedding.from_pretrained(pre_trained_word_vectors)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


###Run training

# Make up some training data
# training_data = [(
#     "the wall street journal reported today that apple corporation made money".split(),
#     "B I I I O O O B I O O".split()
# ), (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )]
#
# word_to_ix = {}
# for sentence, tags in training_data:
#     for word in sentence:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)

# load training data

training_set_sparse_folder = 'C:\\project\\AI\\project_info_extract\\data\\output\\training_set_sparse'
training_data_file = os.listdir(training_set_sparse_folder)
X_train, X_test, y_train, y_test = train_test_split(training_data_file, training_data_file, test_size=0.1, random_state=0)

# tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
tag_to_ix = {
    'B-SHL': 0, 'I-SHL': 1,
    'B-SHS': 2, 'I-SHS': 3,
    'B-CHD': 4, 'I-CHD': 5,
    'B-PRC': 6, 'I-PRC': 7,
    'B-AMT': 8, 'I-AMT': 9,
    'B-CHT': 10, 'I-CHT': 11,
    'B-CPS': 12, 'I-CPS': 13,
    START_TAG: 14, STOP_TAG: 15
}
        # 股东全称        I-SHL    tag type = SHL
        # 股东简称       B- I-SHS
        # 变动截止日期    B- I-CHD
        # 变动价格       B- I-PRC
        # 变动数量       B- I-AMT
        # 变动后持股数    B- I-CHT
        # 变动后持股比例   B- I-CPS

cn_char_vec_file = 'C:\\project\\AI\\data\\chinese_character_vec_1.txt'
cn_char_vec = gensim.models.KeyedVectors.load_word2vec_format(cn_char_vec_file, binary=False, encoding='utf-8')

model = BiLSTM_CRF(tag_to_ix, HIDDEN_DIM, cn_char_vec)  # tag_to_ix, hidden_dim, character_vector
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
# simple test
simple_sample = ''
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        10):  # again, normally you would NOT do 300 epochs, it is toy data
    # ZL: for stochastic gradient descent, the out loop usually
    # set from 1 to 10. if you have a very large training sample
    # you may use small number!!!

    for tag_file in X_train:
        file_path = os.path.join(training_set_sparse_folder, tag_file)
        tagged_content_pair = ContentTagPair.load_from_file(file_path)
        sentence = tagged_content_pair.content_string
        tags = tagged_content_pair.pair_list
        sentence_in = prepare_sequence_with_pre_trained(sentence, cn_char_vec)
    # for sentence, tags in training_data:
    #     # Step 1. Remember that Pytorch accumulates gradients.
    #     # We need to clear them out before each instance
    #     model.zero_grad()
    #
    #     # Step 2. Get our inputs ready for the network, that is,
    #     # turn them into Tensors of word indices.
    #     sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!


def character_is_in_pre_trained_model(char_vec_model, training_data_path):
    errors = list()
    for tag_file in os.listdir(training_data_path):
        in_file_path = os.path.join(training_data_path, tag_file)
        tagged_content_pair = ContentTagPair.load_from_file(in_file_path)
        for char_chinese in tagged_content_pair.content_string:
            try:
                char_vec_model.vocab[char_chinese].index
            except:
                print(char_chinese + ' is not in the vocabulary : ' + tag_file)
                if tag_file not in errors:
                    errors.append(tag_file)
    return errors



# TODO: save the trained model and load it from disk next time.


# TODO: should use cross validation

# word = "whatever"  # for any word in model
# i = model.vocab[word].index
# model.index2word[i] == word  # will be true