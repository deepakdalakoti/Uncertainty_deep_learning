import numpy as np
from torch.autograd import Variable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from util import do_inverse_norm

class MC_dropout():

    def __init__(self, inF, outF, H, drp, lr=1e-4,  problem='regression'):
        self.inF= inF
        self.outF = outF
        self.H = H
        self.problem = problem
        self.lr = lr
        self.drp = drp
        self.model_fn = self.base_model_regression
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        if(problem=='classification'):
            self.model_fn = self.base_model_classification
            self.loss_fn = self.CategoricalCrossentropy()
        return


    def base_model_regression(self):

        init = tf.keras.initializers.glorot_normal()
        
        inputs = Input(shape=(self.inF,))
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(inputs)
        x = Dropout(self.drp)(x)
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        x = Dropout(self.drp)(x)
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        x = Dropout(self.drp)(x)
        x = Dense(self.outF, activation='linear', kernel_initializer=init)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def base_model_classification(self):

        init = tf.keras.initializers.glorot_normal()
        
        inputs = Input(shape=(self.inF,))
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(inputs)
        x = Dropout(self.drp)(x)
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        x = Dropout(self.drp)(x)
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        x = Dropout(self.drp)(x)
        x = Dense(self.outF, activation='softmax', kernel_initializer=init)(x)
        model = keras.Model(inputs=inputs, outputs=x)

        return model

    def loss(self, model, xtrain, ytrain, training):
        ypred = model(xtrain, training=training)
        return self.loss_fn(ytrain, ypred)

    def train_step(self, model, xtrain, ytrain):
        with tf.GradientTape() as tape:
            loss = self.loss(model,xtrain, ytrain, True)
        grad = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return self.optimizer.iterations.numpy(), loss.numpy()

    def train(self, batch_size, epochs, xtrain, ytrain, validation_data=None):
        self.model = self.model_fn()
        train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        train_dataset = train_dataset.shuffle(buffer_size=xtrain.shape[0], reshuffle_each_iteration=True).batch(batch_size)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.999)
        train_loss = []
        valid_loss = []
        for i in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            for x, y in train_dataset:
                step, loss = self.train_step(self.model, xtrain, ytrain)
                epoch_loss_avg.update_state(loss)
            train_loss.append(epoch_loss_avg.result().numpy())
            if(validation_data):
                valid_loss.append(np.mean(self.loss_fn(self.model, validation_data[0], validation_data[1], True).numpy()))
                print("Step {} loss {} valid_loss {}".format(i, epoch_loss_avg.result(), valid_loss[i]))
            else:
                print("Step {} loss {}".format(i, epoch_loss_avg.result()))

        return train_loss, valid_loss

    def predict(self, xdata, y, norm, Nsamp):
        pred_all = np.zeros([Nsamp,xdata.shape[0]])
        for i in range(Nsamp):
            # Predict with dropout using training=True for MC dropout
            pred_all[i,:,None] = self.model(xdata, training=True)

        pred_all = do_inverse_norm(y, pred_all, norm)
        pred_mean = np.mean(pred_all,axis=0)
        pred_std = np.std(pred_all,axis=0)
        return pred_all, pred_mean, pred_std
