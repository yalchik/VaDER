from vader.utils.data_utils import generate_x_w_y
from vader.hp_opt.job.full_optimization_job import FullOptimizationJob


class TestFullOptimizationJob:

    def test_run(self):
        input_data, input_weights, _ = generate_x_w_y(7, 400)
        params_dict = {
            "k": 4,
            "n_hidden": [32, 8],
            "learning_rate": 0.01,
            "batch_size": 16,
            "alpha": 1.0
        }
        seed = 42
        n_consensus = 1
        n_epoch = 10
        n_splits = 2
        n_perm = 10
        job = FullOptimizationJob(input_data, input_weights, params_dict, seed, n_consensus, n_epoch, n_splits, n_perm)
        result = job.run()
        assert result is not None
