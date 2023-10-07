import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Models
import tensorflow as tf
from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier

# Processing....
from tensorflow.keras.utils import to_categorical

class SVM_Model:
    def __init__(self):
        self.model = SVC()

    def _getreport(self):
        return self.model_report
    
    def split_dataset(self, train_percent):
        # Shuffle and Split datasets into triaining and testing datasets
        train, test = np.split(self.df.sample(frac=1), [int(train_percent * len(self.df))])
        
        # Divide each dataset into x_train, x_test and x_valid
        cols = self.df.columns
        self.y_train, self.y_test = train[cols[-1]], test[cols[-1]]
        x_train, x_test = train[cols[:-1]], test[cols[:-1]]
        
        #  Normalize data
        s = StandardScaler()
        self.x_train = s.fit_transform(x_train)
        self.x_test = s.fit_transform(x_test)

        return self.x_train, self.x_test, self.y_train, self.y_test

    
    def train(self, dataset, train_percent):
        self.df = dataset
        x_train, xt, y_train, yt = self.split_dataset(train_percent)

        # Train model with training dataset
        self.model.fit(x_train, y_train)
        self._test_()

    def _test_(self):
        # Test model
        self.y_pred = self.model.predict(self.x_test)
        self.report()

    def report(self):
        self.model_report = classification_report(self.y_test, self.y_pred, output_dict=True)
        print(classification_report(self.y_test, self.y_pred))

# class CNN_Model:
#     def __init__(self):
#         pass

#     def train(self):
#         batch_size = 16
#         nb_classes =4
#         nb_epochs = 5
#         img_rows, img_columns = 200, 200
#         img_channel = 3
#         nb_filters = 32
#         nb_pool = 2
#         nb_conv = 3

#         model = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,input_shape=(200, 200, 3)),
#             tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#             tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
#             tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#             tf.keras.layers.Dropout(0.5),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(128, activation=tf.nn.relu),
#             tf.keras.layers.Dense(4,  activation=tf.nn.softmax)
#         ])
#         model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#         model.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epochs, verbose = 1, validation_data = (X_test, y_test))


class ANN_Model:
    def __init__(self, input_dim, outputdim):
        self.train_percent = train_percent
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(outputdim, activation='softmax')
        ])
    
    def split_dataset(self, train_percent):
        # Shuffle and Split datasets into triaining and testing datasets
        train, test = np.split(self.df.sample(frac=1), [int(train_percent * len(self.df))])
        
        # Divide each dataset into x_train, x_test and x_valid
        cols = self.df.columns
        y_train, y_test = train[cols[-1]], test[cols[-1]]
        x_train, x_test = train[cols[:-1]], test[cols[:-1]]
        
        #  Normalize data
        s = StandardScaler()
        self.x_train = s.fit_transform(x_train)
        self.x_test = s.fit_transform(x_test)

        self.y_train = to_categorical(y_train, 4)
        self.y_test = to_categorical(y_test, 4)

    def _optimizer(self, optim, learning_rate):
        if(lower(optim) == "adam"):
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
            return self.optimizer

        if(lower(optim) == "adamax"):
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate)
            return self.optimizer

    def train(self, dataset, train_percent, optimizer, learning_rate, epochs, batch_size):
        self.df = dataset
        self.split_dataset(train_percent)
        self.model.compile(optimizer=self._optimizer(optimizer, learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.history = self.model.fit(
            self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1
        )

        self.plot_history()

        evl = self.model_evaluation()
        test_res = self._test()

        return {"model evaluation":evl, "model test":tes_res}
    
    def model_evaluation(self):
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1)

        return {"Model Loss":(loss*100), "Accuracy": (accuracy*100)}

    
    def _test(self):
        predictions = self.model.predict(self.x_test)
        rounded_predictions = np.argmax(predictions, axis=1)
        labels = np.argmax(self.y_test, axis=1)

        return {
            "True class":labels, 
            "Predicted class":rounded_predictions, 
            "report": (classification_report(self.y_test, self.y_pred, output_dict=True))
        }
    
    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(self.history.history['loss'], label='loss')
        ax1.plot(self.history.history['val_loss'], label='val_loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Categorical crossentropy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history.history['categorical_accuracy'], label='accuracy')
        ax2.plot(self.history.history['val_categorical_accuracy'], label='val_accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.show()
