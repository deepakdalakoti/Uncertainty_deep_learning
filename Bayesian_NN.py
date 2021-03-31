import numpy as np
from torch.autograd import Variable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from util import do_inverse_norm
import tensorflow_probability as tfp

class Bayesian_net():

    def __init__(self, inF, outF, H, lr=1e-4,  problem='regression'):
        self.inF= inF
        self.outF = outF
        self.H = H
        self.problem = problem
        self.lr = lr
        self.model_fn = self.base_model_regression
        self.train_step = self.train_step_regression
        self.loss_fn = self.loss_reg
        self.pred = self.pred_regression
        if(problem=='classification'):
            self.model_fn = self.base_model_classification
            self.train_step = self.train_step_classification
            self.loss_fn = self.loss_class
            self.pred = self.pred_classification
        return


    def base_model_regression(self):

        inputs = Input(shape=(self.inF,))
        x = tfp.layers.DenseFlipout(self.H,kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=self.kernel_divergence_fn,activation='relu')(inputs)
        x = tfp.layers.DenseFlipout(self.H,kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=self.kernel_divergence_fn,activation='relu')(x)
        x = tfp.layers.DenseFlipout(self.H,kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=self.kernel_divergence_fn,activation='relu')(x)
        mu = tfp.layers.DenseFlipout(self.outF,kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=self.kernel_divergence_fn,activation='linear')(x)
        sigma = tfp.layers.DenseFlipout(self.outF,kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=self.kernel_divergence_fn,activation='softplus')(x)

        model = keras.Model(inputs=inputs, outputs=[mu, sigma])
        return model

    def base_model_classification(self):

        inputs = Input(shape=(self.inF,))
        x = tfp.layers.DenseFlipout(self.H,kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=self.kernel_divergence_fn,activation='relu')(inputs)
        x = tfp.layers.DenseFlipout(self.H,kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=self.kernel_divergence_fn,activation='relu')(x)
        x = tfp.layers.DenseFlipout(self.H,kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=self.kernel_divergence_fn,activation='relu')(x)
        x = tfp.layers.DenseFlipout(self.outF,kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=self.kernel_divergence_fn,activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=x)

        return model

    def kernel_divergence_fn(self, q, p, _):
        return  tfp.distributions.kl_divergence(q, p)

    def loss_reg(self, model, xtrain, ytrain, kl_weight, training):
        # NLL loss
        mu, sigma = model(xtrain, training=training)
        var = sigma + 1e-6
        NLL = tf.math.log(var)*0.5 + 0.5*tf.math.divide(tf.math.square(ytrain-mu),var)  
        total_loss = NLL + kl_weight*sum(model.losses)
        return total_loss
    def loss_class(self, model, xtrain, ytrain, kl_weight, training):
        # 
        ypred = model(xtrain, training=training)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        total_loss = loss_fn(ytrain, ypred) + kl_weight*sum(model.losses)
        return total_loss

    def train_step_regression(self, model, xtrain, ytrain, kl_weight):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(model,xtrain, ytrain, kl_weight, True)
        grad = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return self.optimizer.iterations.numpy(), loss.numpy()

    def train_step_classification(self, model, xtrain, ytrain, kl_weight):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(model, xtrain, ytrain, kl_weight, True)
        grad = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return self.optimizer.iterations.numpy(), loss.numpy()

    def train(self, batch_size, epochs, xtrain, ytrain, kl_weight=1e-3, validation_data=None):
        self.model = self.model_fn()
        train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        train_dataset = train_dataset.shuffle(buffer_size=xtrain.shape[0], reshuffle_each_iteration=True).batch(batch_size)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.999)
        train_loss = []
        valid_loss = []
        for i in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            for x, y in train_dataset:
                step, loss = self.train_step(self.model, xtrain, ytrain, kl_weight)
                epoch_loss_avg.update_state(loss)
            train_loss.append(epoch_loss_avg.result().numpy())
            if(validation_data):
                valid_loss.append(np.mean(self.loss_fn(self.model, validation_data[0], validation_data[1], kl_weight, False).numpy()))
                print("Step {} loss {} valid_loss {}".format(i, epoch_loss_avg.result(), valid_loss[i]))
            else:
                print("Step {} loss {}".format(i, epoch_loss_avg.result()))

        return train_loss, valid_loss


    def pred_regression(self, xdata, y, norm, Nsamp):
        pred_all = np.zeros([Nsamp,xdata.shape[0], 2])
        for i in range(Nsamp):
            mu, sigma = self.model(xdata)
            pred_all[i,:,:]  = np.array([mu, sigma]).T

        # Mean, std as in paper
        pred_mean = np.mean(pred_all[:,:,0],axis=0)
        pred_std = np.sqrt(np.mean(pred_all[:,:,0]**2 + pred_all[:,:,1]+1e-6,axis=0) - pred_mean**2 )
        pred_mean = do_inverse_norm(y, pred_mean, norm)
        pred_std = pred_std*np.std(y)
         
        return pred_all, pred_mean, pred_std

    def pred_classification(self, xdata, y, norm, Nsamp):
        pred_all = np.zeros([Nsamp,xdata.shape[0]])
        for i in range(Nsamp):
            pred_all[i,:] = self.model(xdata)

        pred_all = do_inverse_norm(y, pred_all, norm)
        pred_mean = np.mean(pred_all,axis=0)
        pred_std = np.std(pred_all,axis=0)
        return pred_all, pred_mean, pred_std

    def predict(self, xdata, y, norm, Nsamp):
        pred_all, pred_mean, pred_std = self.pred(xdata, y, norm, Nsamp)
        return pred_all, pred_mean, pred_std
