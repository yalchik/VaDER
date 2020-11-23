from sklearn.base import BaseEstimator, ClusterMixin
from vader.vader import VADER


class VaDERSklearnClustering(BaseEstimator, ClusterMixin):

    def __init__(self, save_path=None, n_hidden=(12, 2), k=3, learning_rate=1e-3, output_activation=None,
                 recurrent=True, batch_size=16, n_epoch=10, verbose=True):
        self.vader = None
        self.verbose = verbose
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.recurrent = recurrent
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.save_path = save_path
        self.k = k

    def fit(self, X):
        vader = VADER(X_train=X, save_path=self.save_path, n_hidden=self.n_hidden, k=self.k,
                      learning_rate=self.learning_rate, output_activation=self.output_activation,
                      recurrent=self.recurrent, batch_size=self.batch_size)

        vader.pre_fit(n_epoch=self.n_epoch, verbose=self.verbose)
        vader.fit(n_epoch=self.n_epoch, verbose=self.verbose)
        self.vader = vader
        return vader

    def predict(self, X):
        return self.vader.predict(X)

    def score(self, X):
        test_loss_dict = self.vader.get_loss(X)
        test_total_loss = test_loss_dict["reconstruction_loss"] + test_loss_dict["latent_loss"]
        return test_total_loss

    def get_params(self, deep=True):
        return {
            "k": self.k,
            "n_epoch": self.n_epoch,
            "batch_size": self.batch_size,
            "recurrent": self.recurrent,
            "output_activation": self.output_activation,
            "learning_rate": self.learning_rate,
            "n_hidden": self.n_hidden,
            "save_path": self.save_path
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
