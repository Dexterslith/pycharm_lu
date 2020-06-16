import tensorflow
import json
import numpy
import random
import nltk
import pickle
from tensorflow import keras

words=[]
docs_x=[]
labels=[]
docs_y=[]
stemmer= nltk.PorterStemmer()

try:
    x
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)
except:
    with open("intents.json") as intent:
        data = json.load(intent)

    for x in data['intents']:
        for y in x['patterns']:
            wrds=nltk.word_tokenize(y)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(x["tag"])

        if x['tag'] not in labels:
            labels.append(x['tag'])
    words=[stemmer.stem(w.lower()) for w in words]
    words=sorted(list(set(words)))
    #print(docs_x)
    labels=sorted(labels)
    training=[]
    output=[]
    #print(classes)
    out_empty=[0 for _ in range(len(labels))]
    #print(out_empty)
    for x,doc in enumerate(docs_x):
        bag=[]
       # print(x,doc)
        wrds = [stemmer.stem(w.lower()) for w in doc if w not in "?"]
       # print(wrds)

        for w in words:
            if w in wrds:
               #print(w,wrds)
                bag.append(1)
            else:
               # print(w,wrds)
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])]=1
       # print(x,doc)
       # print(docs_y[x])
       # print(labels.index(docs_y[x]))
        training.append(bag)
        output.append(output_row)
        with open("data.pickle","wb") as f:
             pickle.dump((words, labels, training, output),f)

    #print(output)
    #print(training)

    training = numpy.array(training)
    output = numpy.array(output)

#print(training)
tensorflow.compat.v1.reset_default_graph()
#print(len(training[0]))
#print(training[0])
#print(numpy.size(training))
#print(len(words))

model=keras.Sequential(
    [
        keras.layers.Dense(48,input_shape=(len(words),)),
        keras.layers.Dense(24,activation="relu"),
        keras.layers.Dense(12,activation="softmax"),
        keras.layers.Dense(len(output[0]),activation="softmax")

    ]
)
keras_model_path = "/tmp/lu_v1"


try:
    keras.models.load_model(keras_model_path)
    print("happy me")
except:
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    model.fit(training,output,epochs=6000)
   # model.save(keras_model_path)


def bag(data,word):
    value=[]
    data=str(data)
    data=nltk.word_tokenize(data)
    data=[stemmer.stem(d.lower()) for d in data]

    for x in word:
        if x in data:
           value.append(1)
        else:
            value.append(0)

    value=numpy.array([value])
    return value

def chat():
    with open("intents.json") as intent:
        d = json.load(intent)



    print("Lets chat:- :) ")
    while True:
        data = input("")
        if data.lower() == "quit":
            break
        else:
            inten=""

            results=model.predict((bag(data,words)))
            tag_val=numpy.argmax(results)
            for x in d['intents']:
                if x['tag'] == labels[tag_val]:
                    inten=x['responses']

            print(random.choice(inten))


chat()