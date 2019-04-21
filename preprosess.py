# -*- coding:utf-8 -*-

import numpy as np
import jieba
import codecs

vocab = {}

def get_vocab(input_file):
    jieba_load_cache = list(jieba.cut("I love you!"))
    global vocab
    with codecs.open(input_file, 'r', encoding='utf-8_sig') as rfile:
        for line in rfile.readlines():
            line = line.strip()
            data = line.split('\t')
            vocab[data[0]] = int(data[1])


def padding_sentence(inputs, max_length):
    result = []
    for data in inputs:
        sentence = [vocab[word] if word in vocab else vocab["UNK"] for word in data]
        if len(sentence) < max_length:
            sentence = sentence + [vocab['PAD']]*(max_length-len(sentence))
        elif len(sentence) > max_length:
            sentence = sentence[:max_length]
        result.append(sentence)
    return result

def seg_sent(sents):
    punc = "！？｡,.!。?？＂＃©@＄％＆＇（）()＊＋，－／/：:-；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    seg_result = []
    for sent in sents:
        seg = list(jieba.cut(sent))
        seg = [token for token in seg if token not in punc]
        seg_result.append(seg)

    return seg_result


def preprocess(query, docs, max_len=32):
    global vocab
    if not len(vocab):
        get_vocab('./resources/vocab/vocab_seg_with_sw')

    query_seg = seg_sent([query])
    docs_seg = seg_sent(docs)

    left_data = padding_sentence(query_seg, max_len)
    left_data = left_data * len(docs)
    right_data = padding_sentence(docs_seg, max_len)

    x_left_data = np.array(left_data)
    x_right_data = np.array(right_data)

    return x_left_data, x_right_data
