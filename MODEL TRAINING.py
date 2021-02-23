from tensorflow import keras
from keras import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

# plt.imshow(x_train[0],cmap="gray")
# plt.show()

# scaler = MinMaxScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# CREATING NEURAL NETWORK

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(200,activation="relu"))
model.add(keras.layers.Dense(200,activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10,activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=5)
history = model.fit(x_train,y_train,epochs=50,validation_split=0.2,callbacks=[early_stop])

pd.DataFrame(history.history).plot()
plt.show()

y_pred = model.predict_classes(x_test)

con_mat = confusion_matrix(y_pred,y_test)
acc_sco = accuracy_score(y_pred,y_test)
# print(con_mat)
print(acc_sco)

model_new = model.save("Digit_Recognizer.hdf5")