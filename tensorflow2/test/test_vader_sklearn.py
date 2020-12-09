from vader.utils import generate_x_w_y
from vader.vader_sklearn import VaDERSklearnClustering
from sklearn.model_selection import GridSearchCV


class TestVaDERSklearnClustering:

    def test_grid_search(self):
        X_train, _, _ = generate_x_w_y(7, 400)
        parameters = {'k': (3, 4, 5)}
        clf = GridSearchCV(VaDERSklearnClustering(), parameters)
        clf.fit(X_train)
        assert clf.cv_results_
