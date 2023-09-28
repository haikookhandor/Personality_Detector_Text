import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from data_preprocess import DataPreprocess
from sklearn.feature_extraction.text import TfidfVectorizer

class Model():
    def __init__(self):
        # Random Forest Regressor
        self.rfr = RandomForestRegressor(bootstrap=True,
         max_features='sqrt',
         min_samples_leaf=1,
         min_samples_split=2,
         n_estimators= 200)
        self.rfc = RandomForestClassifier(max_features='sqrt', n_estimators=110) # Random Forest Classifier
        self.tfidf = TfidfVectorizer(stop_words='english', strip_accents='ascii') # TFIDF Vectorizer

    def fit(self, X, y, regression=True):
        X = self.tfidf.fit_transform(X)
        if regression:
            self.rfr = self.rfr.fit(X, y)
        else:
            self.rfc = self.rfc.fit(X, y)

    def predict(self, X, regression=True):
        X = self.tfidf.transform(X)
        if regression:
            return self.rfr.predict(X)
        else:
            return self.rfc.predict(X)

    def predict_proba(self, X, regression=False):
        X = self.tfidf.transform(X)
        if regression:
            raise ValueError('Not possible to predict the probabilites of regression.')
        else:
            return self.rfc.predict_proba(X)

if __name__ == '__main__':
    traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU'] # Traits to fit models for
    model = Model()

    for trait in traits:
        dp = DataPreprocess() # Data Preprocess object
        X_regression, y_regression = dp.prep_data(trait, regression=True, model_comparison=False) # Get data for regression
        X_categorical, y_categorical = dp.prep_data(trait, regression=False, model_comparison=False) # Get data for classification
        print('Current Fitting trait is' + trait + ' of regression model...')
        model.fit(X_regression, y_regression, regression=True)
        print('Regression done')
        print('Current Fitting trait is' + trait + ' of categorical model...')
        model.fit(X_categorical, y_categorical, regression=False)
        print('Classification done')
        with open('static/' + trait + '_model.pkl', 'wb') as f:
            pickle.dump(model, f) # Save models for future use