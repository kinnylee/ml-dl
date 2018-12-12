import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

seed = 7
numpy.random.seed(seed)

fileName = "../../data/iris.csv"
dataframe = pandas.read_csv(fileName)
dataset = dataframe.values

x = dataset[:, 0:4].astype(float)
y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(y)
encoder_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoder_y)

def baseline_model():
  model = Sequential()
  model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
  model.add(Dense(3, init='normal', activation='sigmoid'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kflod = KFold(n_splits=len(x), shuffle=True, random_state=seed)

results = cross_val_score(estimator, x, dummy_y, cv=kflod)
print(results)
print("Accuracy: %0.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))