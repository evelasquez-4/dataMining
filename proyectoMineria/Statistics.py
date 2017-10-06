from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB  # naive bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier

df=pd.read_csv("features.csv", quotechar='"')

num_features = ['post_comment_count',
                'post_favorite_count','post_score',
                'post_view_count','users_reputation','users_up_votes',
                'users_down_votes','score_prev_acceptans',
                'score_prev_ans','score_prev_comment',
                'score_prev_question','score_prev_favquestion',
                'age_user','title_length','num_block_code','num_i_sentences',
                'num_wh_words','num_y_sentences','tags_popularity','num_tags','code_length']

scaled_features = {}
for each in num_features:
    print(each)
    mean, std = df[each].mean(), df[each].std()
    scaled_features[each] = [mean, std]
    df.loc[:, each] = (df[each] - mean)/std

df=df.iloc[:,1:]
#print(df)

def run_classifier(clf, X, y, num_tests=100):
    scores = []

    for _ in range(num_tests):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_tests)
        clf.fit(X_train, y_train)
        #y_pred = clf.predict(X_test)
        #print(classification_report(y_test, y_pred))
        scores.append(clf.score(X_test, y_test))  # X_test y y_test deben ser definidos previamente

    return np.array(scores)

X = df.iloc[:,1:]
y = df.iloc[:,0]

c1 = ("Decision Tree", DecisionTreeClassifier())
c2 = ("Gaussian NB", GaussianNB())
c3 = ("KNeighbors", KNeighborsClassifier(n_neighbors=20))
dc1 = ("Dummy: stratified", DummyClassifier(strategy="stratified", random_state=0, constant=0))
dc2 = ("Dummy: most_frequent", DummyClassifier(strategy="most_frequent", random_state=0, constant=0))
dc3 = ("Dummy: uniform", DummyClassifier(strategy="uniform", random_state=0, constant=0))

classifiers = [c1, c2, c3, dc1, dc2, dc3]
result_list = []

for name, clf in classifiers:
    print(name)
    accuracys = run_classifier(clf, X, y)
    result_list.append((name, accuracys))

print("+ indica diferencia significativa\n")

for name1, results1 in result_list:
    print("Comparando %s - Accuracy: %.2f" % (name1, results1.mean()))
    print()

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.barh(range(X.shape[1]), importances[indices],
       color="r", xerr=std[indices], align="center")
# If you want to define your own labels,
# change indices to a list of labels on the following line.
plt.yticks(range(X.shape[1]), indices)
plt.ylim([-1, X.shape[1]])
plt.show()




