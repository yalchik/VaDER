from vader.data_utils import generate_x_y_for_nonrecur, generate_x_w_y


class TestDataUtils:

    def test_generate_x_w_y(self):
        X_train, W_train, y_train = generate_x_w_y(7, 400)
        assert X_train.shape == (400, 7, 2)
        assert W_train.shape == (400, 7, 2)
        assert y_train.shape == (400,)

    def test_vader_nonrecur(self):
        X_train, y_train = generate_x_y_for_nonrecur(7, 400)
        assert X_train.shape == (400, 7)
        assert y_train.shape == (400,)
