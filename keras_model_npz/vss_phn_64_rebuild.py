def build_model(npz_filepath):
  
    import tensorflow as tf
    import tensorflow.keras as keras
    import numpy as np
  
    RES = 64
    CHANNELS = 3
    num_filters = 64
    cnn_dropout = 0.3
    
    # Preprocess the input image
    model_input = keras.layers.Input(shape=(RES, RES, CHANNELS))
    processed_input = keras.applications.vgg16.preprocess_input(model_input)
    grey_scale_input = keras.layers.Lambda(
      lambda image: tf.image.rgb_to_grayscale(image), name='grey_scale'
    )(processed_input)
    
    # Block 1
    # conv 1.1
    x = keras.layers.Conv2D(
        num_filters, (3, 3), padding="same", name="block1_conv1",
    )(grey_scale_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # conv 1.2
    x = keras.layers.Conv2D(
        num_filters, (3, 3), padding="same", name="block1_conv2"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # pool 1    
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    x = keras.layers.Dropout(cnn_dropout)(x)
    

    # Block 2
    # conv 2.1
    x = keras.layers.Conv2D(
        num_filters * 2, (3, 3), padding="same", name="block2_conv1"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # conv 2.2    
    x = keras.layers.Conv2D(
        num_filters * 2, (3, 3), padding="same", name="block2_conv2"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # pool 2    
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    x = keras.layers.Dropout(cnn_dropout)(x)

    # Block 3
    # conv 3.1
    x = keras.layers.Conv2D(
        num_filters * 4, (3, 3), padding="same", name="block3_conv1"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # conv 3.2    
    x = keras.layers.Conv2D(
        num_filters * 4, (3, 3), padding="same", name="block3_conv2"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # conv 3.3
    x = keras.layers.Conv2D(
        num_filters * 4, (3, 3), padding="same", name="block3_conv3"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # pool 3
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
    x = keras.layers.Dropout(cnn_dropout)(x)
    

    # Block 4
    # conv 4.1
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block4_conv1"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # conv 4.2
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block4_conv2"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # conv 4.3
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block4_conv3"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # pool 4
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
    x = keras.layers.Dropout(cnn_dropout)(x)
    

    # Block 5
    # conv 5.1
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block5_conv1"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # conv 5.2
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block5_conv2"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # conv 5.3
    x = keras.layers.Conv2D(
        num_filters * 8, (3, 3), padding="same", name="block5_conv3"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # pool 5
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
    x = keras.layers.Dropout(cnn_dropout)(x)
    
    # Get 1D output
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Get additional information
    additional_x = keras.layers.Input(shape=(5,), name="additional_input")
    
    # Merge inputs
    x = keras.layers.concatenate([x, additional_x])
    
    # Dense layers    
    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Activation("relu")(x)
    
    model_output = keras.layers.Dense(1, activation="relu")(x)
    this_model = keras.Model([model_input, additional_x], model_output)
    
    # Compile the model
    this_model.compile(
      keras.optimizers.Adam(
        learning_rate=0.00065536
      ),
      loss="mean_squared_error",
      metrics=[keras.metrics.RootMeanSquaredError()]
    )
    
    fd = np.load(npz_filepath)
    this_model.set_weights([fd[key] for key in fd])
                       
    # this_model.summary()
    return this_model
