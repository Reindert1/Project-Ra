import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import estimator

print("TensorFlow version:", tf.__version__)

SHUFFLE_BUFFER_SIZE = 100

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(109,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='swish'),
    tf.keras.layers.Dense(32, activation='swish'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


def main():
    data = np.load("/Users/rfvis/Documents/GitHub/Project-Ra/data/dataset/full_classification.npy", allow_pickle=True)
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], random_state=69)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    print("Fit model on training data")
    history = model.fit(x_train, y_train, epochs=150, validation_split=0.25, callbacks=[callback])

    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=150)
    print("test loss, test acc:", results)

    print("saving model!")
    model.save('/Users/rfvis/Documents/GitHub/Project-Ra/data/saved_model/new_model.h5')

    print("Plotting results")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print("Plotting loss")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return 0


if __name__ == "__main__":
    main()
