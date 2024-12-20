import tensorflow as tf
from transformers import BertTokenizer, BertModel
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import json
from tqdm import tqdm


bert_tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
bert_model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

with open("data.json") as f:
   dataset = json.load(f)

vectorized_dataset = []
for (question, answer) in zip(dataset['questions'], dataset['answers']):
  vectorized_question = bert_model(**bert_tokenizer(question, return_tensors='pt'))['last_hidden_state'][:, 0, :].detach().numpy()
  vectorized_answer = bert_model(**bert_tokenizer(answer, return_tensors='pt'))['last_hidden_state'][:, 0, :].detach().numpy()
  vectorized_dataset.append([vectorized_question[0], vectorized_answer[0]])

vectorized_dataset = np.array(vectorized_dataset)


model = tf.keras.models.load_model("model.keras")

while True:
  
    
    question=[input("Вы: ")]
    
    emb1 = bert_model(**bert_tokenizer(question, return_tensors='pt'))['last_hidden_state'][:, 0, :].detach().numpy()[0]
    p = []
    for i in tqdm(range(vectorized_dataset.shape[0])):
        emb2 = vectorized_dataset[i, 1]
        emb3 = np.concatenate([emb1, emb2])
        p.append([i, model.predict(np.expand_dims(emb3, axis=0), verbose = False)[0, 0]])
    p = np.array(p)
    answ = np.argmax(p[:, 1])
    print(f"Чат-бот: {dataset['answers'][answ]}")