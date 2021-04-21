import numpy as np
from ensemble import deep_ensemble
from MC_dropout import MC_dropout
from Bayesian_NN import Bayesian_net 
from util import do_normalization
import matplotlib.pyplot as plt

def get_toy_dataset(Npts):
    x = np.linspace(-4,4,Npts)
    y = x**3 + np.random.randn(Npts)*3
    return x,y 


if __name__ == '__main__':
    # Get training data
    x, y = get_toy_dataset(20)
    # Test data
    xtest = np.linspace(-8,8,100)

    # Normalise data
    xN  = do_normalization(x,x,'std')
    yN  = do_normalization(y,y,'std')
    xtestN = do_normalization(xtest, x, 'std')

    xN = xN.reshape(-1,1)
    yN = yN.reshape(-1,1)
    xtestN = xtestN.reshape(-1,1)
    # deep ensemble
    ensemble = deep_ensemble(1,1,10, 5e-3, 5)
    hist, _ = ensemble.train_ensemble(xN, yN, 200, 32)
    _, mean_ensemble, std_ensemble = ensemble.predict(xtestN, y, 'std')

    # MC dropout

    MC = MC_dropout(1,1,10, 0.1, 5e-3)
    hist, _ = MC.train(32, 200, xN, yN)
    _, mean_MC, std_MC = MC.predict(xtestN, y, 'std', 20)

    # Bayesian NN

    BNN = Bayesian_net(1,1,10,5e-3)
    hist, _ = BNN.train(32, 200, xN, yN)
    _, mean_BNN, std_BNN = BNN.predict(xtestN, y, 'std', 20)
    
    print(mean_ensemble.shape, mean_MC.shape, mean_BNN.shape)
    # Plot
    plt.figure(figsize=(10,6))
    plt.subplot(1,3,1)
    plt.plot(xtest, mean_ensemble,'r')
    plt.fill_between(xtest, mean_ensemble[:,0]-3*std_ensemble[:,0], mean_ensemble[:,0] + 3*std_ensemble[:,0])
    plt.scatter(x, y)
    plt.title('Ensemble')
    plt.legend(['Mean prediction', 'Uncertainty', 'training points']) 

    plt.subplot(1,3,2)
    plt.plot(xtest, mean_MC,'r')
    plt.fill_between(xtest, mean_MC-3*std_MC, mean_MC + 3*std_MC)
    plt.scatter(x, y)
    plt.title('MC Dropout')

    plt.subplot(1,3,3)
    plt.plot(xtest, mean_BNN, 'r')
    plt.fill_between(xtest, mean_BNN-3*std_BNN, mean_BNN + 3*std_BNN)
    plt.scatter(x, y)
    plt.title('Bayesian')
    plt.tight_layout()
    plt.savefig('example.png')


