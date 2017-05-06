import numpy as np
import pandas as pd


class multi_regression(object):
    def __init__(self):

        self.x = 1

    weights = []
    feature_matrix = None

    def train(self, data_sframe, features, output, deg=1, step_size=1e-7, tolerance=100):
        self.features = features
        self.deg = deg
        (self.feature_matrix, output_array) = self.get_numpy_data(data_sframe, features, output)
        print self.feature_matrix
        initial_weights = np.zeros(deg * len(features) + 1)

        self.regression_gradient_descent(self.feature_matrix, output_array, initial_weights, step_size, tolerance * deg)

    def get_numpy_data(self, data_sframe, features, output=None):
        poly_frame = pd.DataFrame(index=data_sframe.index)
        poly_frame['constant'] = 1
        np_feats = []
        for feat in features:
            temp, np_feats_temp = self.polynomial_features(data_sframe[feat], self.deg)
            poly_frame = poly_frame.join(temp)
            np_feats.append(np_feats_temp)

        # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
        # this is how you add a constant column to an SFrame
        # add the column 'constant' to the front of the features list so that we can extract it along with the others:
        features = ['constant'] + np_feats  # this is how you combine two lists
        # the following line will convert the features_SFrame into a numpy matrix:
        feature_matrix = poly_frame.as_matrix()
        # assign the column of data_sframe associated with the output to the SArray output_sarray
        if output != None:

            output_sarray = data_sframe[output]
            # the following will convert the SArray into a numpy array by first converting it to a list
            output_array = output_sarray.as_matrix()
            return (feature_matrix, output_array)
        else:
            return feature_matrix

    def polynomial_features(self, feature, degree):
        # assume that degree >= 1
        # initialize the SFrame:
        poly_frame = pd.DataFrame({})
        # and set poly_sframe['power_1'] equal to the passed feature
        poly_frame['power_1'] = feature
        np_feats = []
        np_feats.append('power_1')
        # first check if degree > 1
        if degree > 1:
            # then loop over the remaining degrees:
            # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
            for power in range(2, degree + 1):
                # first we'll give the column a name:
                name = 'power_' + str(power)
                np_feats.append('name')
                # then assign poly_sframe[name] to the appropriate power of feature
                poly_frame[name] = feature ** power
        return poly_frame, np_feats

    def predict(self, feature_matrix):
        features = self.get_numpy_data(feature_matrix, self.features)
        return self.predict_output(features, self.weights)

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
            print weights
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
