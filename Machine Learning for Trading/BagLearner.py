import numpy as np

from scipy import stats


class BagLearner(object):

    def __init__(self, learner=object, kwargs={}, bags = 20, boost = False, verbose = False):

        self.learner = learner
        self.bags = bags
        self.boost = boost
        self.kwargs = kwargs

        #initializing all the learners (Linear regression, Decision tree, Random Tree)
        learners = []
        for i in range(self.bags):
            kwargs_copy = self.kwargs.copy()
            # if self.learner == lrl.LinRegLearner:
            #     if "leaf_size" in kwargs_copy:
            #         kwargs_copy.pop("leaf_size", None)
            #     if "tree" in kwargs_copy:
            #         kwargs_copy.pop("tree",None)

            learners.append(self.learner(**kwargs_copy))

        self.learners = learners

    def author(self):

        return "shusainie3"


    def study_group(self):

        return "None"


    def randomizeddata(self,data_x,data_y):

        #making new arrays to fill x and y data randomly
        rand_indices = np.random.choice(data_x.shape[0], size=data_x.shape[0], replace=True) #with replacement
        randomized_datax = data_x[rand_indices]
        randomized_datay = data_y[rand_indices]
        return randomized_datax, randomized_datay

    def kfold(self, data_x, data_y, current_bag, n_splits):


        n_samples = data_x.shape[0]
        fold_size = n_samples // n_splits

        test_start = current_bag * fold_size
        test_end = test_start + fold_size
        train_x = np.concatenate([data_x[:test_start], data_x[test_end:]], axis=0)
        train_y = np.concatenate([data_y[:test_start], data_y[test_end:]], axis=0)

        return train_x, train_y


    def add_evidence(self,data_x, data_y):
        """

        Parameters
            data_x (numpy.ndarray) – A set of feature values used to train the learner
            data_y (numpy.ndarray) – The value we are attempting to predict given the X data

        Add training data to learners

        """
        #building each model input using randomized data
        for i,learner in enumerate(self.learners*5):
            train_x, train_y = self.kfold(data_x, data_y, current_bag=i, n_splits=i+1)
            learner.add_evidence(train_x,train_y)

    def query(self,points):
        """
        Estimate a set of test points given the model we built.

        Parameters
            points (numpy.ndarray) – A numpy array with each row corresponding to a specific query.

        Returns
            The predicted result of the input data according to the trained model

        Return type
            numpy.ndarray
        """

        # if points.ndim == 1:
        # # Reshape the points array to be 2D if it’s 1D (single sample)
        #     points = points.reshape(1,-1)

        y_preds = np.full((points.shape[0], len(self.learners)), 0, dtype=float)

        for i,learner in enumerate(self.learners):
                y_preds[:,i] = learner.query(points).squeeze()

        return stats.mode(y_preds,axis=1)[0]


if __name__ == "__main__":

    #ran the professor's example initially to make it easier to debug to ensure that it is running as expected

    # data_x = np.array([[0.885,0.725,0.56,0.735,0.61,0.26,0.5,0.32],[0.33,0.39,0.5,0.57,0.63,0.63,0.68,0.78],[9.1,10.9,9.4,9.8,8.4,11.8,10.5,10]])
    # data_x = data_x.T
    # data_y = np.array([4,5,6,5,3,8,7,6])
    # for i in range(10):
    #     learner = BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":1}, bags = 50, boost = False, verbose = False)
    #     learner.add_evidence(data_x, data_y)
    #     Y = learner.query(data_x)
    #     print(Y)

    print("the secret clue is 'zzyzx'")


    # print("The DTlearner seems to be working just fine, I think.....")

