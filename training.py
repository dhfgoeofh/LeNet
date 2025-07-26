# train_and_save_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# ------------------------------
# 1. 공통 데이터 로딩 및 전처리
# ------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# For MLP (flattened input)
x_train_flat = x_train.reshape((-1, 784)).astype('float32') / 255.0
x_test_flat = x_test.reshape((-1, 784)).astype('float32') / 255.0

# For CNN (28x28x1 input)
x_train_cnn = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test_cnn = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# ------------------------------
# 2. MLP with SGD
# ------------------------------
mlp_sgd = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])
mlp_sgd.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(0.01), metrics=['accuracy'])
mlp_sgd.fit(x_train_flat, y_train_cat, epochs=20, batch_size=128, verbose=1)
loss_sgd, acc_sgd = mlp_sgd.evaluate(x_test_flat, y_test_cat)
print("MLP-SGD-MSE Test Accuracy:", acc_sgd)
mlp_sgd.save("models/mlp_sgd_model.h5")

# ------------------------------
# 3. MLP with Adam
# ------------------------------
mlp_adam = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])
mlp_adam.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
mlp_adam.fit(x_train_flat, y_train_cat, epochs=20, batch_size=128, verbose=1)
loss_adam, acc_adam = mlp_adam.evaluate(x_test_flat, y_test_cat)
print("MLP-Adam-MSE Test Accuracy:", acc_adam)
mlp_adam.save("models/mlp_adam_model.h5")

# ------------------------------
# 4. CNN with Adam (LeNet-style)
# ------------------------------
cnn = models.Sequential([
    layers.Conv2D(6, kernel_size=(5,5), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2,2), strides=2),
    layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='valid'),
    layers.MaxPooling2D(pool_size=(2,2), strides=2),
    layers.Conv2D(120, kernel_size=(5,5), activation='relu', padding='valid'),
    layers.Flatten(),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
cnn.fit(x_train_cnn, y_train_cat, epochs=30, batch_size=128, verbose=1)
loss_cnn, acc_cnn = cnn.evaluate(x_test_cnn, y_test_cat)
print("CNN-LeNet5 Test Accuracy:", acc_cnn)
cnn.save("models/cnn_mnist_model.h5")
