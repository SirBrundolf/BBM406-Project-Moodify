import sys
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from LyricsDataExtraction import extract_features
import joblib

# Grid functions create a pipeline and parameters for optimization utilizing grid search
def NBC_grid():
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
    params = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
        'vect__stop_words': ['english', None],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': [1, 1e-1, 1e-2, 1e-3, 1e-4]
    }
    return pipe, params


def SVC_grid():
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC())])
    params = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__C': [0.1, 1, 10, 100, 1000],
        'clf__gamma': [1, 0.1, 0.01, 'scale'],
        'clf__kernel': ['rbf', 'poly', 'linear'],
    }
    return pipe, params

def KNN_grid():
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', KNeighborsClassifier())])
    params = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__use_idf': (True, False),
        'clf__n_neighbors': [3, 5, 8, 12, 15, 20, 29, 40],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan'],
    }
    return pipe, params


def grid_search(dataset_path, model, test_size=0.1, stem=True, lemmatize=False):
    # Split dataset into train and test sets. Consider clusters as labels when related parameter is True
    X_train, X_test, y_train, y_test = extract_features(dataset_path, test_split=test_size, stemming=stem, lemmatization=lemmatize)
    if model == "KNN":
        pipeline, parameters = KNN_grid()
    elif model == "SVC":
        pipeline, parameters = SVC_grid()
    elif model == "NBC":
        pipeline, parameters = NBC_grid()
    else:
        sys.exit("Given model is not valid: " + model)
    grid = GridSearchCV(pipeline, parameters, cv=10, scoring='balanced_accuracy', verbose=3)
    # Save best model
    file_name = model + "_" + "best_model.pkl"
    joblib.dump(grid, file_name)
    grid.fit(X_train, y_train)
    # Show result of grid search
    print("Best Score: ", grid.best_score_)
    print("Best Params: ", grid.best_params_)
    # Test and print report
    y_predict = grid.predict(X_test)
    print(classification_report(y_test, y_predict, digits=3))
    print(confusion_matrix(y_test, y_predict))


def best_model_out(dataset_path, best_model_filepath):
    model = joblib.load(best_model_filepath)
    X_train, X_test, y_train, y_test = extract_features(dataset_path, test_split=0.1, stemming=True, lemmatization=False)
    model.fit(X_train, y_train)
    # Test and print report
    y_predict = model.predict(X_test)
    print(classification_report(y_test, y_predict, digits=3))
    print(confusion_matrix(y_test, y_predict))


# Takes path to the directory involving lyrics files and best model file as command line arguments
if __name__ == "__main__":
    # Prints best model output
    best_model_out(dataset_path=sys.argv[1], best_model_filepath=sys.argv[2])



