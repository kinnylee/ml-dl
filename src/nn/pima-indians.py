from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 7
numpy.random.seed(seed)
fileName = "../../data/pima-indians-diabetes.csv"
dataset = numpy.loadtxt(fileName, delimiter=",", skiprows=1)

x = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, init="uniform", activation="relu"))
model.add(Dense(8, init="uniform", activation="relu"))
model.add(Dense(1, init="uniform", activation="sigmoid"))

# complie model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# train model
# validation_split 指定验证数据的比例
model.fit(x, y, validation_split=0.33, epochs=150, batch_size=10)

# predict
scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# serialize model to json

model_json = model.to_json()
with open('pima-indians-model.json', 'w') as json_file:
  json_file.write(model_json)

model.save_weights('pima-indians.h5')
print('Saved model to disk')

