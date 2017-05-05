import numpy as np


class multi_regression(object):
    def __init__(self):

        self.x = 1

    weights = []
    feature_matrix = None

    def train(self, data_sframe, features, output, step_size=1e-5, tolerance=5):

        (self.feature_matrix, output_array) = self.get_numpy_data(data_sframe, features, output)
        initial_weights = np.zeros(len(features) + 1)

        self.regression_gradient_descent(self.feature_matrix, output_array, initial_weights, step_size, tolerance)

    def get_numpy_data(self, data_sframe, features, output):

        data_sframe['constant'] = 1  # this is how you add a constant column to an SFrame
        # add the column 'constant' to the front of the features list so that we can extract it along with the others:
        features = ['constant'] + features  # this is how you combine two lists
        # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
        features_sframe = data_sframe[features]
        # the following line will convert the features_SFrame into a numpy matrix:
        feature_matrix = features_sframe.as_matrix()
        # assign the column of data_sframe associated with the output to the SArray output_sarray
        output_sarray = data_sframe[output]
        # the following will convert the SArray into a numpy array by first converting it to a list
        output_array = output_sarray.as_matrix()
        print feature_matrix
        print output_array
        return (feature_matrix, output_array)

    def predict(self, feature_matrix):
        return self.predict_output(feature_matrix, self.weights)

    def predict_output(self, feature_matrix, weights):
        # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
        # create the predictions vector by using np.dot()
        predictions = np.dot(feature_matrix, weights)
        return (predictions)

    def feature_derivative(self, errors, feature):

        # Assume that errors and feature are both numpy arrays of the same length (number of data points)
        # compute twice the dot product of these vectors as 'derivative' and return the value
        derivative = 2 * np.dot(errors, feature)
        return (derivative)

    def regression_gradient_descent(self, feature_matrix, output, initial_weights,
                                    step_size, tolerance):

        converged = False
        weights = np.array(initial_weights)  # make sure it's a numpy array
        while not converged:
            # compute the predictions based on feature_matrix and weights using your predict_output() function
            predictions = self.predict_output(feature_matrix, weights)
            # compute the errors as predictions - output
            errors = predictions - output;
            gradient_sum_squares = 0  # initialize the gradient sum of squares
            # while we haven't reached the tolerance yet, update each feature's weight
            for i in range(len(weights)):  # loop over each weight
                # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
                # compute the derivative for weight[i]:
                drev = self.feature_derivative(errors, feature_matrix[:, i])
                # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
                gradient_sum_squares = gradient_sum_squares + drev ** 2
                # subtract the step size times the derivative from the current weight

                weights[i] = weights[i] - step_size * drev
            # compute the square-root of the gradient sum of squares to get the gradient magnitude:
            gradient_magnitude = np.sqrt(gradient_sum_squares)
            if gradient_magnitude < tolerance:
                converged = True
        self.weights = weights
