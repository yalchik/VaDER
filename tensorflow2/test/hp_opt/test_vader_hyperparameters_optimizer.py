import os
import pytest
import shutil
import logging
import multiprocessing as mp
from typing import Final
from vader.utils.data_utils import generate_x_w_y
from vader.hp_opt.interface.abstract_grid_search_params_factory import AbstractGridSearchParamsFactory
from vader.hp_opt.vader_hyperparameters_optimizer import VADERHyperparametersOptimizer


class TestVADERHyperparametersOptimizer:

    OUTPUT_FOLDER: Final[str] = __name__

    class MyParamGridFactory(AbstractGridSearchParamsFactory):

        def get_full_param_dict(self):
            param_dict = {
                "k": [2, 3, 4],
                "n_hidden": [[128, 8], [32, 8], [128, 32]],
                "learning_rate": [0.01],
                "batch_size": [16, 64],
                "alpha": [1.0]
            }
            return param_dict

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        if os.path.exists(self.OUTPUT_FOLDER):
            shutil.rmtree(self.OUTPUT_FOLDER)

        yield

        logging.shutdown()
        if os.path.exists(self.OUTPUT_FOLDER):
            shutil.rmtree(self.OUTPUT_FOLDER)

    def test_run(self):
        X_train, W_train, _ = generate_x_w_y(7, 400)
        optimizer = VADERHyperparametersOptimizer(
            param_grid_factory=self.MyParamGridFactory(),
            seed=None,
            n_repeats=3,
            n_proc=mp.cpu_count(),
            n_sample=3,
            n_consensus=1,
            n_epoch=5,
            n_splits=2,
            n_perm=10,
            output_folder=self.OUTPUT_FOLDER
        )
        optimizer.run(X_train, W_train)
        assert os.path.exists(optimizer.output_pdf_report_file)
        assert os.path.getsize(optimizer.output_pdf_report_file) > 0
