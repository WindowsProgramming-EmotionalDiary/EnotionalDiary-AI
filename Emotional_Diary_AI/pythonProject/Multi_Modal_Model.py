import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def train_emotion_model(data_paths):
    img_size = 224
    num_emotions = 7

    X_data, emotion1_data, y_data = [], [], []
    for path in data_paths:
        data = np.load(path, allow_pickle=True).item()
        imgs = data['imgs']
        emotion1 = data['emotion2']
        labels = data['label']

        X_data.extend(imgs)
        emotion1_data.extend(emotion1)
        y_data.extend(labels)

    X_data = np.array(X_data).astype('float32') / 255.0
    emotion1_data = np.array(emotion1_data).astype('float32')
    y_data = np.array(y_data).astype('int32')

    X_data = np.reshape(X_data, (-1, img_size, img_size, 3))

    X_train, X_test, emotion1_train, emotion1_test, y_train, y_test = train_test_split(
        X_data, emotion1_data, y_data, test_size=0.2, random_state=42
    )

    img_input = Input(shape=(img_size, img_size, 3))

    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=img_input)
    base_model.trainable = False

    emotion1_input = Input(shape=(1,))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = concatenate([x, emotion1_input])
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    output = Dense(num_emotions, activation='softmax')(x)

    model = Model(inputs=[img_input, emotion1_input], outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow([X_train, emotion1_train], y_train, batch_size=32, subset='training')
    validation_generator = datagen.flow([X_train, emotion1_train], y_train, batch_size=32, subset='validation')

    model.fit(train_generator, epochs=20, validation_data=validation_generator)

    loss, accuracy = model.evaluate([X_test, emotion1_test], y_test)
    print(f'테스트 정확도: {accuracy}')

    model.save("emotion_classification_model_resnet50.h5")
    print("모델이 저장되었습니다: emotion_classification_model_resnet50.h5")

    return model

if __name__ == "__main__":
    data_paths = [
        '/emotion/unrest.npy',
        '/emotion/embarrassment.npy',
        '/emotion/anger.npy',
        '/emotion/sandness.npy',
        '/emotion/neutrality.npy',
        '/emotion/happy.npy',
        '/emotion/wound.npy'
    ]
    trained_model = train_emotion_model(data_paths)
