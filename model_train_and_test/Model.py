from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D,Dropout,BatchNormalization
from keras.src.layers import Dense, Activation, Flatten




def model(input_shape,num_classes):
   
      # Build the network model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(64, (1, 1)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(128, (3, 3)))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (1, 1)))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
              
    model.summary()
    return model
    
