import numpy as np
from torch.autograd import Variable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from util import do_inverse_norm, ReduceLROnPlateau

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
        self.model = self.model_fn()
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.999)

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

    @tf.function
    def train_step(self, model, xtrain, ytrain):
        with tf.GradientTape() as tape:
            loss = self.loss(model,xtrain, ytrain, True)
        grad = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss

    def train(self, batch_size, epochs, xtrain, ytrain, validation_data=None):
        train_dataset = tf.data.Dataset.from_tensor_slices((xtrain.astype(np.float32), ytrain.astype(np.float32)))
        train_dataset = train_dataset.shuffle(buffer_size=xtrain.shape[0], reshuffle_each_iteration=True).batch(batch_size)
        #grad_vars = self.model.trainable_weights
        #zero_grads = [tf.zeros_like(w) for w in grad_vars]
        #self.optimizer.apply_gradients(zip(zero_grads, grad_vars)) 
        red_lr = ReduceLROnPlateau(self.optimizer, 0.8, 10, 1e-5) 
        train_loss = []
        valid_loss = []
        epoch_loss_avg = tf.keras.metrics.Mean()
        for i in range(epochs):
            epoch_loss_avg.reset_states()
            for x, y in train_dataset:
                loss = self.train_step(self.model, x, y)
                epoch_loss_avg.update_state(loss)
            train_loss.append(epoch_loss_avg.result())
            red_lr.on_epoch_end(train_loss[-1], i)
            if(validation_data):
                yvalid = self.model(validation_data[0], training=True)
                valid_loss.append(np.mean(self.loss_fn(yvalid, validation_data[1]).numpy()))
                print("Step {} loss {} valid_loss {}".format(i, epoch_loss_avg.result(), valid_loss[i]))
            else:
                print("Step {} loss {}".format(i, epoch_loss_avg.result()))

        return train_loss, valid_loss

    def predict(self, xdata, y, norm, Nsamp):
        pred_all = np.zeros([Nsamp,xdata.shape[0], self.outF])
        for i in range(Nsamp):
            # Predict with dropout using training=True for MC dropout
            pred_all[i,:,:] = self.model(xdata, training=True)

        pred_all = do_inverse_norm(y, pred_all, norm)
        pred_mean = np.mean(pred_all,axis=0)
        pred_std = np.std(pred_all,axis=0)
        return pred_all, pred_mean, pred_std
