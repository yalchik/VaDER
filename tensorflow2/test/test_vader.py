import os
import sys
import pickle
import tensorflow as tf
import vader.utils
from vader import VADER
from vader.utils import generate_x_y_for_nonrecur, generate_x_w_y


class TestVADER:
    @classmethod
    def setup_class(cls):
        tf.config.run_functions_eagerly(False)
        sys.path.append(os.getcwd())
        sys.path.append('tensorflow2')

    def test_vader_recur(self):
        X_train, W_train, y_train = generate_x_w_y(7, 400)
        # Note: y_train is used purely for monitoring performance when a ground truth clustering is available.
        # It can be omitted if no ground truth is available.
        # noinspection PyTypeChecker
        vader = VADER(X_train=X_train, W_train=W_train, y_train=y_train, save_path=None, n_hidden=[12, 2], k=4,
                      learning_rate=1e-3, output_activation=None, recurrent=True, batch_size=16)

        # pre-train without latent loss
        vader.pre_fit(n_epoch=10, verbose=True)
        # train with latent loss
        vader.fit(n_epoch=10, verbose=True)
        # get the clusters
        clustering = vader.cluster(X_train)
        assert any(clustering)
        assert len(clustering) == len(X_train)
        # get the re-constructions
        prediction = vader.predict(X_train)
        assert prediction.shape == X_train.shape
        # compute the loss given the network
        loss = vader.get_loss(X_train)
        assert loss
        assert "reconstruction_loss" in loss
        assert "latent_loss" in loss
        assert loss["reconstruction_loss"] >= 0
        assert loss["latent_loss"] >= 0

    def test_vader_nonrecur(self):
        NUM_OF_TIME_POINTS = 7
        X_train, y_train = generate_x_y_for_nonrecur(NUM_OF_TIME_POINTS, 400)
        # Run VaDER non-recurrently (ordinary VAE with GM prior)
        # noinspection PyTypeChecker
        vader = VADER(X_train=X_train, y_train=y_train, n_hidden=[12, 2], k=2, learning_rate=1e-3,
                      output_activation=None, recurrent=False, batch_size=16)
        # pre-train without latent loss
        vader.pre_fit(n_epoch=10, verbose=True)
        # train with latent loss
        vader.fit(n_epoch=10, verbose=True)
        # get the clusters
        clustering = vader.cluster(X_train)
        assert any(clustering)
        assert len(clustering) == len(X_train)
        # get the re-constructions
        prediction = vader.predict(X_train)
        assert prediction.shape == X_train.shape
        # compute the loss given the network
        loss = vader.get_loss(X_train)
        assert loss
        assert "reconstruction_loss" in loss
        assert "latent_loss" in loss
        assert loss["reconstruction_loss"] >= 0
        assert loss["latent_loss"] >= 0
        # generate some samples
        NUM_OF_GENERATED_SAMPLES = 10
        generated_samples = vader.generate(NUM_OF_GENERATED_SAMPLES)
        assert generated_samples
        assert "clusters" in generated_samples
        assert "samples" in generated_samples
        assert len(generated_samples["clusters"]) == NUM_OF_GENERATED_SAMPLES
        assert generated_samples["samples"].shape == (NUM_OF_GENERATED_SAMPLES, NUM_OF_TIME_POINTS)
