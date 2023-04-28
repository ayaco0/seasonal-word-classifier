from transformers import BertJapaneseTokenizer
from transformers import BertModel
import torch
from numpy import dot
from numpy.linalg import norm
import numpy as np
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

def main():
    word = input("単語を入力してください\n")
    get_season(word)
    # words = ["新学期", "海", "台風", "雪", "ちいかわ"]
    # for word in words:
    #     get_season(word)

def word_to_vector_bert(word):
    input = tokenizer(word, return_tensors="pt")
    outputs = model(**input) 
    # BERTの最終層を取得
    last_hidden_state = outputs.last_hidden_state
    # 最終層から[CLS]と[SEP]のトークンを除いて返す
    return last_hidden_state[0][1]

def cos_sim(v1, v2):
    v1_x = v1.detach().numpy()
    v2_x = v2.detach().numpy()
    return np.dot(v1_x, v2_x) / (np.linalg.norm(v1_x) * np.linalg.norm(v2_x))

def get_season(word):
    # 対象単語の分散表現を取得
    word_vec = word_to_vector_bert(word)
    season_name = ["春", "夏", "秋", "冬"]
    vectors = []
    sims = []
    for season in season_name:
        # 四季の分散表現を取得
        v = word_to_vector_bert(season)
        vectors.append(v)
        # 四季のcos類似度を取得
        s = cos_sim(v,word_vec)
        sims.append(s)

    ans = max(sims)
    print(f"{word}の季語は{season_name[sims.index(ans)]}だよ！")
    # print(ans,sims)

if __name__ == "__main__":
    main()