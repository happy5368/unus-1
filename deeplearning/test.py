import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import cv2
import os

import dl_util_finc as dl

# load the image data
img_dir = './deeplearning/deep_learning_data' 
img_list = os.listdir(img_dir)

# initialize an empty list to hold the preprocessed images
images = [] 

# loop over each image and preprocess it
for img_name in img_list:
    # load the image using opencv
    img = cv2.imread(os.path.join(img_dir, img_name))
    img = cv2.resize(img, (119, 119))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/120.0
    images.append(img)

images = np.array(images) # convert the list of images to a numpy array

# Define the model using TensorFlow's Keras API
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(119,119,3)), # fix me
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(images, labels, epochs=40, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Print the test accuracy
print("Test accuracy:", test_acc)



#####################################################################################


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data() # fix me

train_images = train_images.reshape((50000, 32, 32, 3)) # 32 * 32
test_images = test_images.reshape((10000, 32, 32, 3))
train_images = train_images / 255.0 # 0~1 정규화
test_images = test_images / 255.0

print(train_labels[:10])

# 층을 차례대로 쌓아 tf.keras.Sequential 모델을 만든다
# 은닉층의 활성화 함수는 Relu를 사용하고 출력충은 소프트맥스를 사용
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides = (1, 1), input_shape = (32, 32, 3)))# output feature map의 채널 수 : 32
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25)) # 드롭 아웃을 0.25로 적용하여 전체 노드 중 75%만 사용하도록 함
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))# output feature map의 채널 수 : 64
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25)) # 드롭 아웃을 0.25로 적용하여 전체 노드 중 75%만 사용하도록 함
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))# output feature map의 채널 수 : 128
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25)) # 드롭 아웃을 0.25로 적용하여 전체 노드 중 75%만 사용하도록 함
model.add(tf.keras.layers.Flatten()) # Flatten & FC layer
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.25)) # 드롭 아웃을 0.25로 적용하여 전체 노드 중 75%만 사용하도록 함
model.add(tf.keras.layers.Dense(10, activation='softmax')) # FC layer

# 학습 과정 설정
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 요약 출력
model.summary()

# 모델 학습 수행 (입력데이터x, 입력데이터에 대한 원앤핫 레이블)
history = model.fit(train_images, train_labels, epochs=5, batch_size=10)

plt.figure(figsize=(12, 4))
plt.subplot(1, 1, 1)
# loss는 파란색 점선
plt.plot(history.history['loss'], 'b--', label='loss')
# accuracy는 녹색 실선
plt.plot(history.history['accuracy'], 'g-', label='Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
print('최적화 완료!')

labels = model.predict(test_images)

#  .evaluate() 함수의 반환값: 평가된  [0]loss, [1]accuracy
print("\n Accuracy: %.4f" % (model.evaluate(test_images, test_labels)[1]))