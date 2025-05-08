
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, LSTM
from tensorflow.keras.models import Model

def build_ocr_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    new_shape = (-1, x.shape[1]*x.shape[2], x.shape[3])
    x = Reshape(target_shape=new_shape[1:])(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, x)
    return model
