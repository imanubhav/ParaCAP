from keras.layers import K, Activation
from keras.engine import Layer
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
from keras import optimizers


gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28
embed_size = 300

def get_model():
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(GRU(gru_len,
                          activation='relu',
                          dropout=dropout_p,
                          recurrent_dropout=dropout_p,
                          return_sequences=True))(embed_layer)
    capsule = Capsule(
        num_capsule=Num_capsule,
        dim_capsule=Dim_capsule,
        routings=Routings,
        share_weights=True)(x)

    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    capsule = LeakyReLU()(capsule)

    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(lr=0.001),
        metrics=['accuracy'])
    model.summary()

    return model
#training
model = get_model()
batch_size = 412
epochs = 30

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test))
