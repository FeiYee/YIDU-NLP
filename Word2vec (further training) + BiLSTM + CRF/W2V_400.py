from gensim.models import word2vec
from pytorch_transformers import BertTokenizer
import pickle
import json
import numpy as np

##因为后面要和bert拼接，所以这里是按照bert的分词也就是字级别的分词工具来分，以bert自带的字典来构造word2vec词表

max_len = 512#bert的最大长度
tokenizer_path = "bert-base-uncased"#vocab.txt的位置
sentences = []
file_path = "./data/yidu-s4k/train.txt"#合并两个文件
w2v_path = "word2vec/"

tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=True)#bert的分词器
voacb_path = "word2vec/vocab.txt"#bert的vocab.txt词表
emb_size = 400#word2vec的向量
window_size = 3#word2vec的滑动窗口大小

with open(file_path, encoding="utf-8") as f:
    for line in f:
        line_json = json.loads(line)
        text = line_json["text"]
        #text = "[CLS]" + text + "[SEP]"#因为要和bert匹配，所以在开头和结尾要加上这两个标记，如果是不用bert这句注释。
        if len(text) < 512:
            text += "[PAD]"#因为bert会有pad，所以也训练一下pad的向量
        text = tokenizer.tokenize(text)#bert的分词器分词
        sentences.append(text)
model = word2vec.Word2Vec(sentences, window=window_size, size=emb_size, workers=4, min_count=0)#训练word2vec

w2v = []
vocabs = open(voacb_path, "r", encoding="utf-8").readlines()
#这里按照bert的词表顺序来构造word2vec的词表，这样就可以和bert共用一个输入id
for vocab in vocabs:
    vocab = vocab.replace("\n", "")
    try:
        w2v.append(model[vocab])
    except:
        w2v.append(np.array([0 for _ in range(emb_size)]))#训练语料里未出现的用0初始化
w2v = np.array(w2v, np.float32)#转成numpy array格式
pickle.dump(w2v, open(w2v_path + "w2v_nocls.pkl", "wb"))#保存到本地




