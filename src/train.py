import config.settings as conf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import datasets
from joblib import dump


class Train():
    def __init__(self):
        pass

    def load_data(self):
        '''
        Load data from sklearn datasets
        '''
        diabetes = datasets.load_diabetes()
        return diabetes.data, diabetes.target

    def create_dataframe(self, X, y, cols):
        '''
        Create pandas dataframe from given features and target
        '''
        Y = np.array([y]).transpose()
        d = np.concatenate((X, Y), axis=1)
        return pd.DataFrame(d, columns=cols)

    def split_data(self, data, train_size=0.7, test_size=0.3):
        '''
        Split data
        '''
        np.random.seed(conf.SEED)

        # Split the data into training and test sets.
        train, test = train_test_split(data, train_size=train_size, test_size=test_size)
        return train, test

    def train(self, train, alpha=0.05, l1_ratio=0.05):
        '''
        Train the model
        '''

        # The predicted column is "progression" which is a quantitative
        # measure of disease progression one year after baseline
        train_x = train.drop(["progression"], axis=1)
        train_y = train[["progression"]]

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=conf.SEED)
        return lr.fit(train_x, train_y)

    @staticmethod
    def eval_metrics(actual, pred):
        '''
        Get evaluate metrics
        '''
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def evaluate(self, model, test):
        '''
        Evaluate the model using a test data
        '''
        test_x = test.drop(["progression"], axis=1)
        test_y = test[["progression"]]

        # Make predictions
        predicted_qualities = model.predict(test_x)

        # Evaluate the predictions
        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

        # Print out ElasticNet model metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (conf.ALPHA, conf.L1_RATIO))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

    def predict(self, model, data):
        '''
        Make predictions
        '''
        return model.predict(data)

    def persist_model(self, model, path):
        '''
        Persist the model to a given path
        '''
        dump(model, path)
