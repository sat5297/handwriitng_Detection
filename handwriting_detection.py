import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)
model.fit(x_train,y_train,epochs=3)
model.fit(x_train,y_train,epochs=3)
Epoch 1/3
60000/60000 [==============================] - 2s 41us/sample - loss: 0.2668 -
acc: 0.9212
Epoch 2/3
60000/60000 [==============================] - 2s 34us/sample - loss: 0.1098 -
acc: 0.9664
Epoch 3/3
60000/60000 [==============================] - 2s 36us/sample - loss: 0.0742 -
acc: 0.9770
val_loss,val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc)
model.save('num_recogo.model')
new_model = tf.keras.models.load_model('num_recogo.model')
predictions = model.predict([x_test])
print(np.argmax(predictions[0]))
Ouput:
7
plt.imshow(x_test[0],cmap = plt.cm.binary)
plt.show()
