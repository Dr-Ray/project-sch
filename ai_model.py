import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
import numpy as np 

class AIModel:
    def __init__(self, learning_rate, input_dim, train={}):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy']) 
        if(train):
            self.pre_train(train, 2)

    def pre_train(self, training_data, silent=0):
        training_data['y'] = to_categorical(training_data['y'], 4)
        self.model.fit(training_data['x'], training_data['y'], epochs=100, batch_size=32, validation_split=0.2, verbose=silent)
    
    def train(self, x_train, y_train, epochs, batch_size, validation_split, verbose=0, plot_hist=False):    
        y_train = to_categorical(y_train, 4)

        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

        if(plot_hist):
            self.plot_history(history)
    
    def model_test(self, x_test, y_test):
        y_test = to_categorical(y_test, 4)
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)

        return loss, accuracy
    
    def predict(self, new_val):
        predictions = self.model.predict(new_val)
        rounded_predictions = np.argmax(predictions, axis=1)

        return rounded_predictions
    
    def categorize_output(self, inputs):
        categories = np.argmax(inputs, axis=1)

        return categories

    
    def plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history.history['loss'], label='loss')
        ax1.plot(history.history['val_loss'], label='val_loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Categorical crossentropy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history.history['categorical_accuracy'], label='accuracy')
        ax2.plot(history.history['val_categorical_accuracy'], label='val_accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.show()


