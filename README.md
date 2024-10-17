lenet_model=tf.keras.Sequential([
    InputLayer(input_shape=(CONFIGURATIONS["IM_SIZE"],CONFIGURATIONS["IM_SIZE"],3)),
    Rescaling(1./255),
    layers.Conv2D(filters=CONFIGURATIONS["N_FILTERS"],kernel_size=CONFIGURATIONS["KERNEL_SIZE"],strides=CONFIGURATIONS["N_STRIDE"],activation="relu",kernel_regularizer=l2(CONFIGURATIONS["REGULARIZATION_RATE"])),
    BatchNormalization(),
    layers.MaxPooling2D(pool_size=CONFIGURATIONS["POOL_SIZE"],strides=CONFIGURATIONS["N_STRIDE"]*2),
    Dropout(rate=CONFIGURATIONS["DROPOUT_RATE"]),
    layers.Conv2D(filters=CONFIGURATIONS["N_FILTERS"]*2+4,kernel_size=CONFIGURATIONS["KERNEL_SIZE"],strides=CONFIGURATIONS['N_STRIDE'],activation="relu",kernel_regularizer=l2(CONFIGURATIONS["REGULARIZATION_RATE"])),
    BatchNormalization(),
    layers.MaxPooling2D(pool_size=CONFIGURATIONS["POOL_SIZE"],strides=CONFIGURATIONS["N_STRIDE"]*2),
    
    Flatten(),
    layers.Dense(CONFIGURATIONS["DENSE_1"],activation="relu",kernel_regularizer=l2(CONFIGURATIONS['REGULARIZATION_RATE'])),
    BatchNormalization(),
    Dropout(rate=CONFIGURATIONS["DROPOUT_RATE"]),
    layers.Dense(CONFIGURATIONS["DENSE_2"],activation="relu",kernel_regularizer=l2(CONFIGURATIONS['REGULARIZATION_RATE'])),
    BatchNormalization(),
    layers.Dense(CONFIGURATIONS["DENSE_3"],activation="sigmoid")
]
    
)
lenet_model.summary()
