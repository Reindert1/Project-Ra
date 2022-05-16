import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
print("TensorFlow version:", tf.__version__)

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(None, 64, 14)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def main():
    data = np.load("/Users/rfvis/Documents/GitHub/Project-Ra/data/r4-c7_nucleus.npy", allow_pickle=True)

    x_train, x_val, y_train, y_val = train_test_split(data[:, :-1],
                                                      data[:, -1],
                                                      random_state=42)
    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=2,
        validation_data=(x_val, y_val),
    )

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_val, y_val, batch_size=128)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(x_val[:50])
    print("predictions shape:", predictions.shape)

    return 0


if __name__ == "__main__":
    main()
