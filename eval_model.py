import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, SGDRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

class ModelEvaluator():
    def __init__(self, X, y, trait):
        self.X = X
        self.y = y
        self.trait = trait
        self.models_dict = { # Models to compare
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(max_features='sqrt', n_estimators=110),
            'MultinomialNB': MultinomialNB(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor' : RandomForestRegressor(
                 bootstrap=True,
                 max_features='sqrt',
                 min_samples_leaf=1,
                 min_samples_split=2,
                 n_estimators= 200),
            'Ridge': Ridge(),
            'SGDRegressor': SGDRegressor(),
        }
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.hyperparameters = {
        'RandomForestClassifier': {'max_features': 'sqrt', 'n_estimators': 110},
        }
    # Hyperparameter tuning
    def tune_hyperparameters(self, model):
        traits = ['O', 'C', 'E', 'A', 'N']
        trait_best_params_dict = {}
        for trait in traits:
            if model == 'RandomForestRegressor':

                n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)]
                max_features = ['auto', 'sqrt']
                max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
                max_depth.append(None)
                
                # Create the random grid of parameters like n_estimators, max_features, max_depth
                random_grid = {'n_estimators': n_estimators,
                               'max_features': max_features,
                               }


                # Use the random grid to search for best hyperparameters
                rf = RandomForestRegressor()
                # Do gridsearch in order to get the best parameters
                random_forest_GS = GridSearchCV(estimator=rf, param_grid=random_grid, cv=5)

                # Fit the random search model
                random_forest_GS.fit(self.X, self.y)
                print('Current Personality ' + trait + ' is best params: ' )
                for k, v in random_forest_GS.best_params_:
                    print (k + ': ' + v)
                trait_best_params_dict[trait] = random_forest_GS.best_params_

        return trait_best_params_dict

    # Compare model performance
    def compare_scores(self, models, regression=False):
        print('Model performance for trait ' + self.trait + ' prediction:' + '\n')
        # Initialize lists to store scores
        accuracy_scores = []
        f1_scores = []

        for model_name in models:
            model = self.models_dict[model_name]
            model.fit(self.X_train, self.y_train)

            print(model_name + ": ") # Print model name to keep track of progress

            if regression:
                y_pred = model.predict(self.X_test)
                y_true = self.y_test
                mse = -np.mean(cross_validate(model, self.X_test, self.y_test, scoring='neg_mean_squared_error', cv=10)['test_score'])
                print('The MSE is: ' + str(mse) + '\n')
            else:
                accuracy_score = np.mean(cross_validate(model, self.X_test, self.y_test, cv=10)['test_score'])
                accuracy_scores.append(accuracy_score)
                print('Accuracy score: ' + str(accuracy_score) + '\n')
                f_score = np.mean(cross_validate(model, self.X_test, self.y_test, scoring='f1', cv=10)['test_score'])
                f1_scores.append(f_score)
                print('The F1 score is: ' + str(f_score) + '\n')

        if regression:
            pass
        else:
            best_accuracy_score = max(accuracy_scores)
            best_accuracy_model = models[accuracy_scores.index(best_accuracy_score)]
            print(
                'Best Accuracy score: ' + str(best_accuracy_score) + '\n' +
                'Model: ' + best_accuracy_model + '\n' + '\n'
            )
            best_f1_score = max(f1_scores)
            best_f1_model = models[f1_scores.index(best_f1_score)]
            print(
                'Best F1 score: ' + str(best_f1_score) + '\n' +
                'Model: ' + best_f1_model + '\n'
            )