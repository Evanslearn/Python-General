import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model


def plot_sample_predictions(i0):
    test_width = 4
    lowlim = i0 - test_width
    uplim = i0 + test_width + 1
    print(f'Index {i0} given. We will try to take the indexes in the interval [{lowlim, uplim}].')
    plot_counter = 0
    sample_results = []

    for number in range(lowlim, uplim):
        i = number % len(x_val)  # (We make sure that the sample number is not larger than our validation dataset)
        if number > (len(x_val) - 1):
            print(
                f'\nValue index {number} > {len(x_val) - 1} (validation dataset length). Instead, the value {i} will be used')
        elif number < 0:
            print(f'\nValue index {number} < 0. Instead, the value {i} will be used (the max index, plus the number)')
        else:
            print(f'\nValue index {number} <= {len(x_val) - 1} (validation dataset length). It will be used')

        prediction = np.argmax(predictions[i])
        print(f'Our predictions are predictions[i] = {predictions[i]}')
        print(f'The max value corresponds to class {prediction}.\nThe label value is {y_val[i]}.')
        if prediction == y_val[i]:
            print(f'The sample prediction was correct.')
        else:
            print(f'The sample prediction was misclassified.')
        pyplot.subplot(330 + 1 + plot_counter)
        pyplot.imshow(x_val[i], cmap=pyplot.get_cmap('gray'))
        sample_results.append([prediction, y_val[i]])
        plot_counter += 1

    print(f'\nOur results in the form [prediction, y_val[i]] -> {sample_results}')
    pyplot.show()

# Loading the dataset gives us 4 variables
(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()


# We cast our tensors to a different type
# We will transform our pixel data (0-255) to 0-1
# The label/class (y) is supposed to be an int
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x, y


def create_dataset(xs, ys, n_classes=10):
    # One hot vector is a way to make labels into numbers. The labels for mnist fashion are:
    # 0 T-shirt/top
    # 1 Trouser
    # 2 Pullover
    # 3 Dress
    # 4 Coat
    # 5 Sandal
    # 6 Shirt
    # 7 Sneaker
    # 8 Bag
    # 9 Ankle boot
    # For example. if we have only the first 3 clothes' labels:
    # Shirt -> 0
    # Trousers -> 1
    # Pullover -> 2
    # But we cannot simply use this label to number relationship and be done with it.
    # Instead, we transform each label to a vector with dimensionality equal to the number of labels so:
    # Shirt -> [1, 0, 0]
    # Trousers -> [0, 1, 0]
    # Pullover -> [0, 0, 1]
    # These are the so called one-hot vectors
    ys = tf.one_hot(ys, depth=n_classes)

    # from_tensor_slices gives us a new data entry for each x-y pair
    # map() executes the function we created that manipulates our data set
    # shuffle randomly shuffles our data
    # batch(N) organizes our data in batches of size (N)
    return tf.data.Dataset.from_tensor_slices((xs, ys)) \
        .map(preprocess) \
        .shuffle(len(ys)) \
        .batch(128)


train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)


class KerasModels:

    def kSeq():
        model = keras.Sequential([
            keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dense(units=192, activation='relu'),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=10, activation='softmax')
        ])
        return model


ModelObject = KerasModels

MyModel = ModelObject.kSeq()
MyModel.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = MyModel.fit(
    train_dataset.repeat(),
    epochs=10,
    steps_per_epoch=500,
    validation_data=val_dataset.repeat(),
    validation_steps=2

)

# Store the predictions (predictions on the validation set)
predictions = MyModel.predict(val_dataset)
# Our model outputs a probability distribution about how likely it is for each clothing type to be shown on an image.
# To make our classification, We take the one with the highest probability


# Sample predictions (predictions[i])
# ----- Change the value of i0 below this line -----
i0 = 0
# ----- Change the value of i0 above this line -----
plot_sample_predictions(i0)



#tf.keras.utils.plot_model(MyModel, to_file='model.png')
#print(f'prediction -> {prediction}')