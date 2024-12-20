import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization 
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TFAutoModel
import numpy as np

import data

rubert_model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(rubert_model_name)
rubert_model = TFAutoModel.from_pretrained(rubert_model_name, from_pt=True)

dataset = []
for i in range(len(data.questions)):
    q_encoded = tokenizer(data.questions[i:i+1], padding=True, truncation=True, return_tensors="tf")
    a_encoded = tokenizer(data.answers[i:i+1], padding=True, truncation=True, return_tensors="tf")

    q_emb = rubert_model(q_encoded).last_hidden_state[:, 0, :].numpy() 
    a_emb = rubert_model(a_encoded).last_hidden_state[:, 0, :].numpy() 

    dataset.append([np.array(q_emb[0]), np.array(a_emb[0])])

dataset = np.array(dataset)

embeddings = dataset[:, 1]  
np.save("dataset_embeddings.npy", embeddings)

np.save("answers.npy", np.array(data.answers))

X, Y = [], []
for i in range(dataset.shape[0]):
    for j in range(dataset.shape[0]):
        X.append(np.concatenate([dataset[i, 0, :], dataset[j, 1, :]], axis=0))
        if i == j:
            Y.append(1)
        else:
            Y.append(0)
X = np.array(X)
Y = np.array(Y)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(1536, )))
model.add(tf.keras.layers.Dense(100, activation="selu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='pr', name='auc')])
model.fit(X, Y, epochs=10000, class_weight={0:1, 1:np.sqrt(Y.shape[0])-1})


model.save('model.keras')
