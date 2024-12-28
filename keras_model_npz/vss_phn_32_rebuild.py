

def build_model(npz_filepath):
  
    import tensorflow as tf
    import tensorflow.keras as keras
    import numpy as np
  
    RES = 32
    CHANNELS = 3
    num_filters = 32
    cnn_dropout = 0.4
    
    # Preprocess the input image
    model_input = keras.layers.Input(shape=(RES, RES, CHANNELS), name="input_1")
    processed_input = keras.applications.vgg16.preprocess_input(model_input)
    grey_scale_input = keras.layers.Lambda(
      lambda image: tf.image.rgb_to_grayscale(image), name='grey_scale'
    )(processed_input)
    
    # Block 1
    # conv 1.1
    x = keras.layers.Conv2D(
        num_filters, (3, 3), padding="same", name="block1_conv1",
    )(grey_scale_input)
    x = keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = keras.layers.Activation("relu", name="activation")(x)
    
    # conv 1.2
    x = keras.layers.Conv2D(
        num_filters, (3, 3), padding="same", name="block1_conv2"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_1")(x)
    x = keras.layers.Activation("relu", name="activation_1")(x)
    
    # pool 1    
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    x = keras.layers.Dropout(cnn_dropout, name="dropout")(x)

    # Block 2
    # conv 2.1
    x = keras.layers.Conv2D(
        num_filters * 2, (3, 3), padding="same", name="block2_conv1"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_2")(x)
    x = keras.layers.Activation("relu", name="activation_2")(x)
    
    # conv 2.2    
    x = keras.layers.Conv2D(
        num_filters * 2, (3, 3), padding="same", name="block2_conv2"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_3")(x)
    x = keras.layers.Activation("relu", name="activation_3")(x)
    
    # pool 2    
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    x = keras.layers.Dropout(cnn_dropout, name="dropout_1")(x)

    # Block 3
    # conv 3.1
    x = keras.layers.Conv2D(
        num_filters * 4, (3, 3), padding="same", name="block3_conv1"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_4")(x)
    x = keras.layers.Activation("relu", name="activation_4")(x)
    
    # conv 3.2    
    x = keras.layers.Conv2D(
        num_filters * 4, (3, 3), padding="same", name="block3_conv2"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_5")(x)
    x = keras.layers.Activation("relu", name="activation_5")(x)
    
    # conv 3.3
    x = keras.layers.Conv2D(
        num_filters * 4, (3, 3), padding="same", name="block3_conv3"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_6")(x)
    x = keras.layers.Activation("relu", name="activation_6")(x)
    
    # pool 3
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
    x = keras.layers.Dropout(cnn_dropout, name="dropout_2")(x)

    # Block 4
    # conv 4.1
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block4_conv1"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_7")(x)
    x = keras.layers.Activation("relu", name="activation_7")(x)
    
    # conv 4.2
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block4_conv2"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_8")(x)
    x = keras.layers.Activation("relu", name="activation_8")(x)
    
    # conv 4.3
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block4_conv3"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_9")(x)
    x = keras.layers.Activation("relu", name="activation_9")(x)
    
    # pool 4
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
    x = keras.layers.Dropout(cnn_dropout, name="dropout_3")(x)
    

    # Block 5
    # conv 5.1
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block5_conv1"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_10")(x)
    x = keras.layers.Activation("relu", name="activation_10")(x)
    
    # conv 5.2
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block5_conv2"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_11")(x)
    x = keras.layers.Activation("relu", name="activation_11")(x)
    
    # conv 5.3
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block5_conv3"
    )(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_12")(x)
    x = keras.layers.Activation("relu", name="activation_12")(x)
    
    # pool 5
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
    x = keras.layers.Dropout(cnn_dropout, name="dropout_4")(x)
    
    # Get 1D output
    x = keras.layers.GlobalMaxPooling2D(name="global_max_pooling2d")(x)
    
    # Get additional information
    additional_x = keras.layers.Input(shape=(5,), name="additional_input")
    
    # Merge inputs
    x = keras.layers.concatenate([x, additional_x], name="concatenate")
    
    # Dense layers    
    x = keras.layers.Dense(256, name="dense")(x)
    x = keras.layers.Dropout(0.2, name="dropout_5")(x)
    x = keras.layers.Activation("relu", name="activation_13")(x)
    
    model_output = keras.layers.Dense(1, activation="relu", name="dense_1")(x)
    this_model = keras.Model([model_input, additional_x], model_output)
    
    # Compile the model
    this_model.compile(
      keras.optimizers.Adam(
        learning_rate=0.00032768
      ),
      loss="mean_squared_error",
      metrics=[keras.metrics.RootMeanSquaredError()]
    )
    
    fd = np.load(npz_filepath)
    this_model.set_weights([fd[key] for key in fd])
    
    # layer_name = ["input_1", "tf.__operators__.getitem", "tf.nn.bias_add", "grey_scale", "block1_conv1", "batch_normalization", "activation", "block1_conv2", "batch_normalization_1", "activation_1", "block1_pool", "dropout", "block2_conv1", "batch_normalization_2", "activation_2", "block2_conv2", "batch_normalization_3", "activation_3", "block2_pool", "dropout_1", "block3_conv1", "batch_normalization_4", "activation_4", "block3_conv2", "batch_normalization_5", "activation_5", "block3_conv3", "batch_normalization_6", "activation_6", "block3_pool", "dropout_2", "block4_conv1", "batch_normalization_7", "activation_7", "block4_conv2", "batch_normalization_8", "activation_8", "block4_conv3", "batch_normalization_9", "activation_9", "block4_pool", "dropout_3", "block5_conv1", "batch_normalization_10", "activation_10", "block5_conv2", "batch_normalization_11", "activation_11", "block5_conv3", "batch_normalization_12", "activation_12", "block5_pool", "dropout_4", "global_max_pooling2d", "additional_input", "concatenate", "dense", "dropout_5", "activation_13", "dense_1"]
    # 
    # for i, layer in enumerate(this_model.layers):
    #     if (i == 1 or i == 2):
    #         continue
    #     if layer_name[i] != layer.name:
    #         raise ValueError(f'Layer name "{layer.name}" does not match the desired layer name "{layer_name[i]}"!')
    #                    
    # this_model.summary()
    return this_model
