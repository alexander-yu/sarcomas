from keras import backend as K
from keras.layers import average, AveragePooling2D, concatenate, Conv2D, Conv3D, Dense, Flatten, Input, Reshape, MaxPooling2D, Dropout, maximum, Lambda, Activation
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD

PATCH_HEIGHT = 28
PATCH_WIDTH = 28

k_1 = 5
k_2 = 2

p_1 = 4
p_2 = 2

k_g = k_1 + k_2 + p_1 + p_2 - 3

d_i = 0
d_l = k_1 + p_1 - 2
d_mf = k_1 + k_2 + p_1 + p_2 - 4


def get_type_1_model(summary=False):
    K.clear_session()

    ct_input = Input(shape=(PATCH_HEIGHT, PATCH_WIDTH, 1))
    pet_input = Input(shape=(PATCH_HEIGHT, PATCH_WIDTH, 1))

    x = concatenate([ct_input, pet_input])
    x = Reshape((PATCH_HEIGHT, PATCH_WIDTH, 2, 1))(x)
    x = Conv3D(16, (2, 2, 2), activation='relu')(x)
    x = Reshape((27, 27, 16))(x)
    x = Conv2D(36, (2, 2), activation='relu')(x)
    x = Conv2D(64, (2, 2), activation='relu')(x)
    x = Conv2D(144, (2, 2), activation='relu')(x)
    x = AveragePooling2D((23, 23))(x)
    x = Flatten()(x)
    x = Dense(864, activation='relu')(x)
    x = Dense(288, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[ct_input, pet_input], outputs=output)

    model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()

    return model


def get_type_2_model(summary=False):
    K.clear_session()

    ct_input = Input(shape=(PATCH_HEIGHT, PATCH_WIDTH, 1))
    pet_input = Input(shape=(PATCH_HEIGHT, PATCH_WIDTH, 1))

    ct_model = Conv2D(16, (2, 2), activation='relu')(ct_input)
    ct_model = Conv2D(36, (2, 2), activation='relu')(ct_model)
    ct_model = Conv2D(64, (2, 2), activation='relu')(ct_model)
    ct_model = Conv2D(144, (2, 2), activation='relu')(ct_model)
    ct_model = AveragePooling2D((23, 23))(ct_model)
    ct_model = Flatten()(ct_model)

    pet_model = Conv2D(16, (2, 2), activation='relu')(pet_input)
    pet_model = Conv2D(36, (2, 2), activation='relu')(pet_model)
    pet_model = Conv2D(64, (2, 2), activation='relu')(pet_model)
    pet_model = Conv2D(144, (2, 2), activation='relu')(pet_model)
    pet_model = AveragePooling2D((23, 23))(pet_model)
    pet_model = Flatten()(pet_model)

    x = concatenate([ct_model, pet_model])
    x = Dense(864, activation='relu')(x)
    x = Dense(288, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[ct_input, pet_input], outputs=output)

    model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()
    
    return model


def get_type_3_model(summary=False):
    K.clear_session()

    ct_input = Input(shape=(PATCH_HEIGHT, PATCH_WIDTH, 1))
    pet_input = Input(shape=(PATCH_HEIGHT, PATCH_WIDTH, 1))

    ct_model = Conv2D(16, (2, 2), activation='relu')(ct_input)
    ct_model = Conv2D(36, (2, 2), activation='relu')(ct_model)
    ct_model = Conv2D(64, (2, 2), activation='relu')(ct_model)
    ct_model = Conv2D(144, (2, 2), activation='relu')(ct_model)
    ct_model = AveragePooling2D((23, 23))(ct_model)
    ct_model = Flatten()(ct_model)
    ct_model = Dense(864, activation='relu')(ct_model)
    ct_model = Dense(288, activation='relu')(ct_model)
    ct_model = Dense(1, activation='sigmoid')(ct_model)

    pet_model = Conv2D(16, (2, 2), activation='relu')(pet_input)
    pet_model = Conv2D(36, (2, 2), activation='relu')(pet_model)
    pet_model = Conv2D(64, (2, 2), activation='relu')(pet_model)
    pet_model = Conv2D(144, (2, 2), activation='relu')(pet_model)
    pet_model = AveragePooling2D((23, 23))(pet_model)
    pet_model = Flatten()(pet_model)
    pet_model = Dense(864, activation='relu')(pet_model)
    pet_model = Dense(288, activation='relu')(pet_model)
    pet_model = Dense(1, activation='sigmoid')(pet_model)

    predictions = average([ct_model, pet_model])

    model = Model(inputs=[ct_input, pet_input], outputs=predictions)

    model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()
    
    return model


def get_single_modality_model(summary=False):
    K.clear_session()
    
    model = Sequential()
    model.add(Conv2D(16, (2, 2), activation='relu', input_shape=(PATCH_HEIGHT, PATCH_WIDTH, 1)))
    model.add(Conv2D(36, (2, 2), activation='relu'))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(Conv2D(144, (2, 2), activation='relu'))
    model.add(AveragePooling2D((23, 23)))
    model.add(Flatten())
    model.add(Dense(864, activation='relu'))
    model.add(Dense(288, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()
    
    return model


def get_stream_model(big_patch_dim, small_patch_dim, n_feature_maps=2, mode=None, summary=False, maxout=False, dropout=False):
    K.clear_session()
        
    k_f = big_patch_dim - k_g + 1

    if mode in ['ct', 'pet']:
        x = Input(shape=(big_patch_dim, big_patch_dim, 1))
        model_input = x
    else:
        ct_input = Input(shape=(big_patch_dim, big_patch_dim, 1))
        pet_input = Input(shape=(big_patch_dim, big_patch_dim, 1))
        model_input = [ct_input, pet_input]
        x = concatenate(model_input, axis=-1)
    
    if maxout:
        conv1_local = maximum([Conv2D(64, (k_1, k_1))(x) for _ in range(n_feature_maps)])
    else:
        conv1_local = Conv2D(64, (k_1, k_1), activation='relu')(x)
    pool1_local = MaxPooling2D((p_1, p_1), strides=(1, 1))(conv1_local)
    if dropout:
        pool1_local = Dropout(0.2)(pool1_local)
    
    if maxout:
        conv2_local = maximum([Conv2D(64, (k_2, k_2))(pool1_local) for _ in range(n_feature_maps)])
    else:
        conv2_local = Conv2D(64, (k_2, k_2), activation='relu')(pool1_local)
    pool2_local = MaxPooling2D((p_2, p_2), strides=(1, 1))(conv2_local)
    if dropout:
        pool2_local = Dropout(0.2)(pool2_local)

    if maxout:
        conv1_global= maximum([Conv2D(160, (k_g, k_g))(x) for _ in range(n_feature_maps)])
    else:
        conv1_global = Conv2D(160, (k_g, k_g), activation='relu')(x)
    if dropout:
        conv1_global = Dropout(0.2)(conv1_global)
    
    #combine = Flatten()(concatenate([pool2_local, conv1_global], axis=-1))
    #output = Dense(small_patch_dim * small_patch_dim, activation='sigmoid')(combine)
    combine = concatenate([pool2_local, conv1_global], axis=-1)
    output = Conv2D(small_patch_dim * small_patch_dim, (k_f, k_f), activation='sigmoid')(combine)
    
    if small_patch_dim > 1:
        output = Reshape((small_patch_dim, small_patch_dim, 1))(output)
    else:
        output = Reshape((1,))(output)

    model = Model(inputs=model_input, outputs=output)

    model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()
    return model


def get_two_path_cascade_input(stream_model, n_feature_maps=2, mode=None, summary=False, maxout=False, dropout=False):
    K.clear_session()
    
    k_f = PATCH_HEIGHT - k_g + 1

    stream_model = stream_model(2 * PATCH_HEIGHT - d_i, PATCH_HEIGHT,
                                mode=mode, n_feature_maps=n_feature_maps, maxout=maxout, dropout=dropout)
    stream_model.trainable = False
    stream_model.load_weights(f'input_stream_{mode}.h5' if mode is not None else 'input_stream.h5')
    
    if mode in ['ct', 'pet']:
        model_input = Input(shape=(2 * PATCH_HEIGHT - d_i, 2 * PATCH_WIDTH - d_i, 1))
        x = model_input
        stream_output = stream_model(x)
    else:
        ct_input = Input(shape=(2 * PATCH_HEIGHT - d_i, 2 * PATCH_WIDTH - d_i, 1))
        pet_input = Input(shape=(2 * PATCH_HEIGHT - d_i, 2 * PATCH_WIDTH - d_i, 1))
        model_input = [ct_input, pet_input]
        x = concatenate(model_input, axis=-1)
        stream_output = stream_model([ct_input, pet_input])
    
    h = (PATCH_HEIGHT - d_i) // 2
    w = (PATCH_WIDTH - d_i) // 2
    trim = d_i % 2 == 1
        
    x = Lambda(lambda x: x[:, h+trim:-h, w+trim:-w, :])(x)
    x = concatenate([x, stream_output], axis=-1)
    
    if maxout:
        conv1_local = maximum([Conv2D(64, (k_1, k_1))(x) for _ in range(n_feature_maps)])
    else:
        conv1_local = Conv2D(64, (k_1, k_1), activation='relu')(x)
    pool1_local = MaxPooling2D((p_1, p_1), strides=(1, 1))(conv1_local)
    if dropout:
        pool1_local = Dropout(0.2)(pool1_local)
        
    if maxout:
        conv2_local = maximum([Conv2D(64, (k_2, k_2))(pool1_local) for _ in range(n_feature_maps)])
    else:
        conv2_local = Conv2D(64, (k_2, k_2), activation='relu')(pool1_local)
    pool2_local = MaxPooling2D((p_2, p_2), strides=(1, 1))(conv2_local)
    if dropout:
        pool2_local = Dropout(0.2)(pool2_local)

    if maxout:
        conv1_global = maximum([Conv2D(160, (k_g, k_g))(x) for _ in range(n_feature_maps)])
    else:
        conv1_global = Conv2D(160, (k_g, k_g), activation='relu')(x)
    if dropout:
        conv1_global = Dropout(0.2)(conv1_global)
    
    #combine = Flatten()(concatenate([pool2_local, conv1_global], axis=-1))
    #output = Dense(1, activation='sigmoid')(combine)
    combine = concatenate([pool2_local, conv1_global], axis=-1)
    output = Conv2D(1, (k_f, k_f), activation='sigmoid')(combine)
    output = Reshape((1,))(output)

    model = Model(inputs=model_input, outputs=output)

    model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()
    return model


def get_two_path_cascade_local(stream_model, n_feature_maps=2, mode=None, summary=False, maxout=False, dropout=False):
    K.clear_session()
    
    k_f = PATCH_HEIGHT - k_g + 1
    
    stream_model = stream_model(2 * PATCH_HEIGHT - d_l, PATCH_HEIGHT - d_l,
                                mode=mode, n_feature_maps=n_feature_maps, maxout=maxout, dropout=dropout)
    stream_model.trainable = False
    stream_model.load_weights(f'local_stream_{mode}.h5' if mode is not None else 'local_stream.h5')
    
    if mode in ['ct', 'pet']:
        x = Input(shape=(2 * PATCH_HEIGHT - d_l, 2 * PATCH_WIDTH - d_l, 1))
        model_input = x
        stream_output = stream_model(x)
    else:
        ct_input = Input(shape=(2 * PATCH_HEIGHT - d_l, 2 * PATCH_WIDTH - d_l, 1))
        pet_input = Input(shape=(2 * PATCH_HEIGHT - d_l, 2 * PATCH_WIDTH - d_l, 1))
        model_input = [ct_input, pet_input]
        x = concatenate([ct_input, pet_input], axis=-1)
        stream_output = stream_model([ct_input, pet_input])
    
    h = (PATCH_HEIGHT - d_l) // 2
    w = (PATCH_WIDTH - d_l) // 2
    trim = d_l % 2 == 1
        
    x = Lambda(lambda x: x[:, h+trim:-h, w+trim:-w, :])(x)
    
    if maxout:
        conv1_local = maximum([Conv2D(64, (k_1, k_1))(x) for _ in range(n_feature_maps)])
    else:
        conv1_local = Conv2d(64, (k_1, k_1), activation='relu')(x)
    pool1_local = MaxPooling2D((p_1, p_1), strides=(1, 1))(conv1_local)
    if dropout:
        pool1_local = Dropout(0.2)(pool1_local)
    
    pool1_local = concatenate([pool1_local, stream_output], axis=-1)
    
    if maxout:
        conv2_local = maximum([Conv2D(64, (k_2, k_2))(pool1_local) for _ in range(n_feature_maps)])
    else:
        conv2_local = Conv2D(64, (k_2, k_2), activation='relu')(pool1_local)
    pool2_local = MaxPooling2D((p_2, p_2), strides=(1, 1))(conv2_local)
    if dropout:
        pool2_local = Dropout(0.2)(pool2_local)

    if maxout:
        conv1_global= maximum([Conv2D(160, (k_g, k_g))(x) for _ in range(n_feature_maps)])
    else:
        conv1_global = Conv2D(160, (k_g, k_g), activation='relu')(x)
    if dropout:
        conv1_global = Dropout(0.2)(conv1_global)
    
    combine = concatenate([pool2_local, conv1_global], axis=-1)
    output = Conv2D(1, (k_f, k_f), activation='sigmoid')(combine)
    output = Reshape((1,))(output)

    model = Model(inputs=model_input, outputs=output)

    model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()
    return model


def get_two_path_cascade_mf(stream_model, n_feature_maps=2, mode=None, summary=False, maxout=False, dropout=False):
    K.clear_session()
    
    k_f = PATCH_HEIGHT - k_g + 1
    
    stream_model = stream_model(2 * PATCH_HEIGHT - d_mf, PATCH_HEIGHT - d_mf,
                                mode=mode, n_feature_maps=n_feature_maps, maxout=maxout, dropout=dropout)
    stream_model.trainable = False
    stream_model.load_weights(f'mf_stream_{mode}.h5' if mode is not None else 'mf_stream.h5')
    
    if mode in ['ct', 'pet']:
        x = Input(shape=(2 * PATCH_HEIGHT - d_mf, 2 * PATCH_WIDTH - d_mf, 1))
        model_input = x
        stream_output = stream_model(x)
    else:
        ct_input = Input(shape=(2 * PATCH_HEIGHT - d_mf, 2 * PATCH_WIDTH - d_mf, 1))
        pet_input = Input(shape=(2 * PATCH_HEIGHT - d_mf, 2 * PATCH_WIDTH - d_mf, 1))
        model_input = [ct_input, pet_input]
        x = concatenate(model_input, axis=-1)
        stream_output = stream_model([ct_input, pet_input])
    
    h = (PATCH_HEIGHT - d_mf) // 2
    w = (PATCH_WIDTH - d_mf) // 2
    trim = d_mf % 2 == 1
        
    x = Lambda(lambda x: x[:, h+trim:-h, w+trim:-w, :])(x)
    
    if maxout:
        conv1_local = maximum([Conv2D(64, (k_1, k_1))(x) for _ in range(n_feature_maps)])
    else:
        conv1_local = Conv2D(64, (k_1, k_1), activation='relu')(x)
    pool1_local = MaxPooling2D((p_1, p_1), strides=(1, 1))(conv1_local)
    if dropout:
        pool1_local = Dropout(0.2)(pool1_local)
    
    if maxout:
        conv2_local = maximum([Conv2D(64, (k_2, k_2))(pool1_local) for _ in range(n_feature_maps)])
    else:
        conv2_local = Conv2D(64, (k_2, k_2), activation='relu')(pool1_local)
    pool2_local = MaxPooling2D((p_2, p_2), strides=(1, 1))(conv2_local)
    if dropout:
        pool2_local = Dropout(0.2)(pool2_local)

    if maxout:
        conv1_global= maximum([Conv2D(160, (k_g, k_g))(x) for _ in range(n_feature_maps)])
    else:
        conv1_global = Conv2D(160, (k_g, k_g), activation='relu')(x)
    if dropout:
        conv1_global = Dropout(0.2)(conv1_global)

    combine = concatenate([pool2_local, conv1_global, stream_output], axis=-1)
    output = Conv2D(1, (k_f, k_f), activation='sigmoid')(combine)
    output = Reshape((1,))(output)

    model = Model(inputs=model_input, outputs=output)

    model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()
    return model