import pandas as pd
from datetime import datetime
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from classifiers import CoTrainingClassifier


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def time_now():
    return str(datetime.now().strftime('%Y_%m_%d__%H_%M_%S'))


def run(k, labels, classifier, sfm=False, thr='1.0'):
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for j in range(k):
        print(j)
        data_train = pd.read_csv('data/norm_sets/train/train_data%d.csv' % j)
        data_test = pd.read_csv('data/norm_sets/test/test_data%d.csv' % j)

        x_train = data_train[labels]
        x_test = data_test[labels]
        y_train = data_train['hateLabel']
        y_test = list(data_test['hateLabel'])

        if sfm:
            print()
            print(x_train.shape)
            print(x_test.shape)

            classifier.fit(x_train, y_train)

            model = SelectFromModel(classifier, threshold='%s*mean' % thr, prefit=True)
            x_train = model.transform(x_train)
            x_test = model.transform(x_test)

            print(x_train.shape)
            print(x_test.shape)
            print()

        classifier.fit(x_train, y_train)
        y_pred = list(classifier.predict(x_test))

        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))

    mean_eval = {
        'accuracy': mean(accuracy),
        'precision': mean(precision),
        'recall': mean(recall),
        'f1_score': mean(f1)
    }

    return mean_eval


# source: http://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
    for i in range(len(y_hat)):
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
    for i in range(len(y_hat)):
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return TP, FP, TN, FN


def write_labels(file, labels, i=1):
    file.write('Labels %d:\n' % i)
    for label in labels:
        file.write('\t%s\n' % label)


def cotrain(
        labels_1, labels_2, name_1, name_2, classifier_1, classifier_2, iterations=1, u=75, p=1, n=1, k_i=30,
        poly=False, matrix=False, conf_mat=False, k_fold=10):

    if matrix:
        file_str = 'data/outfiles/cotrain/final_cot/%d_%d_%d_%d_out%s.txt' % (u, p, n, k_i, time_now())
    else:
        file_str = 'data/outfiles/cotrain/cotrain_output_10fold_%s.txt' % time_now()

    outfile = open(file_str, 'w')

    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    outfile.write('CoTraining - %s\n' % name_1)
    outfile.write('CoTraining - %s\n\n' % name_2)

    for idx in range(k_fold):
        train_file = 'data/cotrain/train/cotrain_data_train_%d.csv' % idx
        test_file = 'data/cotrain/test/cotrain_data_test_%d.csv' % idx
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        if poly:
            polynomial = PolynomialFeatures(2)

        for i in range(iterations):
            y_test = test_data['hateLabel'].values
            y = train_data['hateLabel'].values

            if poly:
                X1 = polynomial.fit_transform(train_data[labels_1].values)
                X2 = polynomial.fit_transform(train_data[labels_2].values)
                X1_test = polynomial.fit_transform(test_data[labels_1].values)
                X2_test = polynomial.fit_transform(test_data[labels_2].values)
            else:
                X1 = train_data[labels_1].values
                X2 = train_data[labels_2].values
                X1_test = test_data[labels_1].values
                X2_test = test_data[labels_2].values

            lg_co_clf = CoTrainingClassifier(classifier_1, classifier_2, u=u, p=p, n=n, k=k_i)
            lg_co_clf.fit(X1, X2, y)
            y_pred = lg_co_clf.predict(X1_test, X2_test)

            if conf_mat:
                TP, FP, TN, FN = perf_measure(y_test, y_pred)

                outfile.write('\n')
                outfile.write('TP:\t%d\n' % TP)
                outfile.write('FP:\t%d\n' % FP)
                outfile.write('TN:\t%d\n' % TN)
                outfile.write('FN:\t%d\n' % FN)
                outfile.write('\n')
                outfile.write('%s\n' % y_test)
                outfile.write('%s\n' % y_pred)

            accuracy += accuracy_score(y_test, y_pred)
            precision += precision_score(y_test, y_pred)
            recall += recall_score(y_test, y_pred)
            temp_f1 = f1_score(y_test, y_pred)
            f1 += temp_f1

            print(i)

    outfile.write('Accuracy:\t%s\n' % (accuracy / (iterations * k_fold)))
    outfile.write('Precision:\t%s\n' % (precision / (iterations * k_fold)))
    outfile.write('Recall:\t\t%s\n' % (recall / (iterations * k_fold)))
    outfile.write('F1 score:\t%s\n\n' % (f1 / (iterations * k_fold)))

    write_labels(outfile, labels_1, 1)
    write_labels(outfile, labels_2, 2)

    outfile.close()


def test_all_classifiers(k, labels):
    csvfile = open('data/outfiles/final/all_classifiers_output_%s.csv' % time_now(), 'w')

    csvfile.write('classifier,accuracy,precision,recall,f1\n')

    classifiers = {
        'dc': DummyClassifier(),
        'rfc': RandomForestClassifier(
            n_estimators=40,
            n_jobs=-1,
            max_features='log2'),
        'abc': AdaBoostClassifier(
            base_estimator=ExtraTreesClassifier(
                n_estimators=200,
                n_jobs=-1,
                max_features='sqrt',
                min_samples_leaf=1,
                max_depth=5),
            n_estimators=200),
        'bc': BaggingClassifier(n_jobs=-1),
        'etc': ExtraTreesClassifier(
            n_estimators=200,
            n_jobs=-1,
            max_features='sqrt',
            min_samples_leaf=1,
            max_depth=5),
        'gbc': GradientBoostingClassifier(),
        'lr': LogisticRegression(n_jobs=-1),
        'pac': PassiveAggressiveClassifier(n_jobs=-1),
        'rc': RidgeClassifier(),
        'mnb': MultinomialNB(),
        'bnb': BernoulliNB(),
        'knc': KNeighborsClassifier(n_jobs=-1),
        'nc': NearestCentroid(),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.0002,
            learning_rate='invscaling',
            max_iter=300),
        'dtc': DecisionTreeClassifier(),
        'etc2': ExtraTreeClassifier(),
        'svc_rbf': SVC(),
    }

    for classifier_key, classifier in sorted(classifiers.items()):
        mean_eval = run(k=k, labels=labels, classifier=classifier)

        csvfile.write('%s,%s,%s,%s,%s\n' % (
            classifier_key,
            mean_eval['accuracy'],
            mean_eval['precision'],
            mean_eval['recall'],
            mean_eval['f1_score']))

        print('%s DONE' % classifier_key)

    csvfile.write('\n')
    write_labels(csvfile, labels)

    csvfile.close()


def matrix_test_extra_trees(
        k, labels, combinations, run_times=10, n_estimators_list=[40],
        criterion_list=['gini'], max_features_list=['log2'], min_samples_split_list=[2]):

    csvfile = open('data/outfiles/tree_matrix_output_%s.csv' % time_now(), 'w')

    csvfile.write('n_estimators,criterion,max_features,min_samples_split,accuracy,precision,recall,f1\n')

    counter = 1

    for n_estimators in n_estimators_list:
        for criterion in criterion_list:
            for max_features in max_features_list:
                for min_samples_split in min_samples_split_list:
                    for i in range(len(labels), len(labels) + 1):
                        for comb in combinations[i]:
                            mean_accuracy = 0
                            mean_precision = 0
                            mean_recall = 0
                            mean_f1_score = 0

                            for j in range(0, run_times):
                                classifier = ExtraTreesClassifier(
                                    n_estimators=n_estimators,
                                    n_jobs=-1,
                                    max_features=max_features,
                                    criterion=criterion,
                                    min_samples_split=min_samples_split)

                                mean_eval = run(k=k,
                                                labels=list(comb),
                                                classifier=classifier)

                                mean_accuracy += mean_eval['accuracy']
                                mean_precision += mean_eval['precision']
                                mean_recall += mean_eval['recall']
                                mean_f1_score += mean_eval['f1_score']

                            csvfile.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                                n_estimators,
                                criterion,
                                max_features,
                                min_samples_split,
                                (mean_accuracy / run_times),
                                (mean_precision / run_times),
                                (mean_recall / run_times),
                                (mean_f1_score / run_times)))

                            print(counter)

                            counter += 1

    csvfile.close()


def matrix_test_gradient_boost(
        k, labels, combinations, run_times=10, loss_list=['deviance'], learning_rate_list=[0.1],
        n_estimators_list=[100], max_depth_list=[3], criterion_list=['friedman_mse'], max_features_list=[None]):

    csvfile = open(
        'data/outfiles/boost_matrix_output_%s.csv' % time_now(), 'w')

    csvfile.write('loss,learning_rate,n_estimators,max_depth,criterion,max_features,accuracy,precision,recall,f1\n')

    counter = 1

    for loss in loss_list:
        for learning_rate in learning_rate_list:
            for n_estimators in n_estimators_list:
                for max_depth in max_depth_list:
                    for criterion in criterion_list:
                        for max_features in max_features_list:
                            for i in range(len(labels), len(labels) + 1):
                                for comb in combinations[i]:
                                    mean_accuracy = 0
                                    mean_precision = 0
                                    mean_recall = 0
                                    mean_f1_score = 0

                                    for j in range(0, run_times):
                                        classifier = GradientBoostingClassifier(
                                            loss=loss,
                                            learning_rate=learning_rate,
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            criterion=criterion,
                                            max_features=max_features)

                                        mean_eval = run(k=k, labels=list(comb), classifier=classifier)

                                        mean_accuracy += mean_eval['accuracy']
                                        mean_precision += mean_eval['precision']
                                        mean_recall += mean_eval['recall']
                                        mean_f1_score += mean_eval['f1_score']

                                    csvfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                                        loss,
                                        learning_rate,
                                        n_estimators,
                                        max_depth,
                                        criterion,
                                        max_features,
                                        (mean_accuracy / run_times),
                                        (mean_precision / run_times),
                                        (mean_recall / run_times),
                                        (mean_f1_score / run_times)))

                                    print(counter)

                                    counter += 1

    csvfile.close()


def get_feature_importance(k, labels, name, classifier):
    csvfile = open('data/outfiles/%s_features_importance_%s.csv' % (name, time_now()), 'w')

    csvfile.write('feature,importance\n')

    run(k=k, labels=labels, classifier=classifier)

    importances = classifier.feature_importances_

    for label, importance in list(zip(labels, importances)):
        csvfile.write('%s,%s\n' % (label, importance))

    csvfile.close()
