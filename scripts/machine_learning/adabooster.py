#!usr/bin/env python3

"""
    Boosting algorithm using Adaboost
"""

__author__ = "Reindert"
__version__ = 0.1


# Imports
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class adaboost:
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self):
        x_train, x_val, y_train, y_val = train_test_split(self.dataset[:, :-1], data_array[:, -1], random_state=0)
        for label in dataset.columns:
            print("Running:", label)
            dataset[label] = LabelEncoder().fit(self.dataset[label]).transform(self.dataset[label])

        X = x_train.drop(['Label'], axis=1)
        Y = y_train['Label']

        print("Initializing booster...")
        AdaBoost = AdaBoostClassifier(n_estimators=5, learning_rate=0.1, algorithm='SAMME')

        print("Fitting...")
        AdaBoost.fit(X, Y)

        prediction = AdaBoost.score(X, Y)

        print('The accuracy is: ', prediction * 100, '%')


def main():
    data_array = np.load("../../data/r4-c7_nucleus.npy", allow_pickle=True)
    print("Shape:", data_array.shape)

    dataset = pd.DataFrame(data_array, columns=['Pixel_val1', 'Pixel_val2', 'Pixel_val3', 'Pixel_val4', 'Pixel_val5',
                                                'Pixel_val6', 'Pixel_val7', 'Pixel_val8', 'Pixel_val9', 'Gauss1',
                                                'Gauss2', 'Gauss3', 'Gauss4', 'Label'])

    adaboost(dataset)

    return 0


if __name__ == "__main__":
    exitcode = main()
    sys.exit(exitcode)

