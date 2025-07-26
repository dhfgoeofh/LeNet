import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.datasets import mnist
import os

# 1. 모델 로딩
model_mlp_sgd = tf.keras.models.load_model("mlp_sgd_model.h5")
model_mlp_adam = tf.keras.models.load_model("mlp_adam_model.h5")
model_cnn = tf.keras.models.load_model("cnn_mnist_model.h5")

# 2. MNIST 데이터 로딩 (예시 시각화용)
(_, _), (x_test, y_test) = mnist.load_data()
x_test_flat = x_test.reshape((-1, 784)).astype('float32') / 255.0
x_test_cnn = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# 3. 0~9 각 숫자 하나씩 예측 후 시각화
def show_test_predictions():
    indices = [np.where(y_test == i)[0][0] for i in range(10)]  # 각 숫자 하나씩
    imgs = x_test[indices]
    x_flat = x_test_flat[indices]
    x_cnn = x_test_cnn[indices]

    preds_sgd = np.argmax(model_mlp_sgd.predict(x_flat), axis=1)
    preds_adam = np.argmax(model_mlp_adam.predict(x_flat), axis=1)
    preds_cnn = np.argmax(model_cnn.predict(x_cnn), axis=1)

    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.subplot(3, 10, i + 1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(f"True: {y_test[indices[i]]}")
        plt.axis('off')

        plt.subplot(3, 10, 10 + i + 1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(f"MLP-SGD\nPred: {preds_sgd[i]}")
        plt.axis('off')

        plt.subplot(3, 10, 20 + i + 1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(f"CNN\nPred: {preds_cnn[i]}")
        plt.axis('off')
    plt.suptitle("Prediction Results (Top: Ground Truth, Middle: MLP-SGD, Bottom: CNN)", fontsize=16)
    plt.tight_layout()
    plt.show()

# 4. 손글씨 입력을 통한 예측
def predict_drawn_image():
    drawing = np.zeros((280, 280), dtype=np.uint8)
    drawing_copy = drawing.copy()
    drawing_flag = False

    def draw(event, x, y, flags, param):
        nonlocal drawing_flag, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_flag = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing_flag:
            cv2.circle(drawing, (x, y), 10, (255,), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing_flag = False

    cv2.namedWindow("Draw a Digit (Press ESC to Predict)")
    cv2.setMouseCallback("Draw a Digit (Press ESC to Predict)", draw)

    while True:
        cv2.imshow("Draw a Digit (Press ESC to Predict)", drawing)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

    # 전처리
    img = cv2.resize(drawing, (28, 28))
    img_norm = img.astype("float32") / 255.0
    img_input_flat = (1.0 - img_norm).reshape(1, 784)
    img_input_cnn = (1.0 - img_norm).reshape(1, 28, 28, 1)

    pred_sgd = model_mlp_sgd.predict(img_input_flat)
    pred_adam = model_mlp_adam.predict(img_input_flat)
    pred_cnn = model_cnn.predict(img_input_cnn)

    def plot_bar(pred, title):
        plt.bar(range(10), pred[0])
        plt.xticks(range(10))
        plt.title(f"{title}: {np.argmax(pred)} (conf: {np.max(pred):.2f})")

    # 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plot_bar(pred_sgd, "MLP-SGD")

    plt.subplot(1, 4, 3)
    plot_bar(pred_adam, "MLP-Adam")

    plt.subplot(1, 4, 4)
    plot_bar(pred_cnn, "CNN")

    plt.tight_layout()
    plt.show()

# 실행
if __name__ == "__main__":
    print("▶ 1. MNIST 테스트셋 예측 결과 시각화")
    show_test_predictions()

    print("▶ 2. 손글씨 숫자를 마우스로 그리고 ESC로 종료 후 추론")
    predict_drawn_image()
