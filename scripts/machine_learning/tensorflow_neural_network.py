import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def main():
    data_loc = ...

    with np.load(data_loc) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # train
    model.fit(train_dataset, epochs=10)

    # eval
    model.evaluate(test_dataset, verbose=2)

    return 0


if __name__ == "__main__":
    main()
