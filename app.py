import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import matplotlib.pyplot as plt

# cargamos las 4 combinaciones de las compuertas XOR
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

# y estos son los resultados que se obtienen, en el mismo orden
target_data = np.array([[0], [1], [1], [0]], "float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

history = model.fit(training_data, target_data, epochs=1000)

plt.figure()
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(history.history["loss"])
plt.legend()

plt.show()

data_test = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], "float32")
# evaluamos el modelo
scores = model.evaluate(data_test, target_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.predict(data_test).round())
