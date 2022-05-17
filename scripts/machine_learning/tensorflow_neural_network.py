import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import estimator

print("TensorFlow version:", tf.__version__)

SHUFFLE_BUFFER_SIZE = 100

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(109,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(32)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def main():
    data = np.load("/Users/rfvis/Documents/GitHub/Project-Ra/data/dataset/full_classification.npy", allow_pickle=True)
    print(np.shape(data))
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1],
                                                        data[:, -1],
                                                        random_state=42)

    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        validation_data=(x_test, y_test),
    )

    print("Using model:\n", model)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=150)
    print("test loss, test acc:", results)

    predictions = model.predict(x_test)

    return 0


if __name__ == "__main__":
    main()
