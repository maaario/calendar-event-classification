import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from stemmsk import stem

###################
# SIMPLE NP UTILS #
###################

def normalize(x):
    """ 
    Normalize dataset to mean 0 and sd 1. 
    Use before PCA
    """
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0)
    return (x - m) / s

def count_freqs(x):
    """
    Returns a list of pairs. One pair corresponds to a distinct `value` - (value, count of elements with value).
    """
    bc = np.bincount(x)
    ii = np.nonzero(bc)[0]
    return list(zip(ii,bc[ii]))

##################
# PLOTTING UTILS #
##################

def label_points(ax, x, y, labels):
    """
    This function renders boxes with labels into a plot.
    Provide axis of figure, xy coordinates of data points and point labels.
    """
    for px, py, label in zip(x, y, labels):
        if label:
            ax.annotate(label, xy=(px, py), xytext=(15, 20), textcoords='offset points', 
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))


def category_legend(category_to_int):
    """
    Adds a legend to a plot based on `category_to_int` mapping 
    """
    category_ints = category_to_int.values()
    norm = mpl.colors.Normalize(vmin=min(category_ints), vmax=max(category_ints))
    color_map = mpl.cm.ScalarMappable(norm=norm)

    patches = []
    for category, color_int in sorted(category_to_int.items(), key=lambda x: x[1]):
        patch = mpl.patches.Patch(color=color_map.to_rgba(color_int), label=category)
        patches.append(patch)
        
    plt.legend(handles=patches)            

############
# ML UTILS #
############

def evaluate_model(model, X, y, splits):
    """
    Trains and validates model using k-fold splits. Prints out training and testing mean accuracy.
    Example:
        from sklearn import svm
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        splits = list(kf.split(X))
        svc = svm.SVC(C=1kernel='rbf')
        evaluate_model(svc, X, y, splits)
    """
    k = len(splits)
    train_acc, test_acc = [], []
    train_f1, test_f1 = [], []
    
    
    for train_index, test_index in splits:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        
        train_acc.append(model.score(X_train, y_train))
        test_acc.append(model.score(X_test, y_test))
        
        train_f1.append(f1_score(y_train, model.predict(X_train), average='macro'))
        test_f1.append(f1_score(y_test, model.predict(X_test), average='macro'))
        
    print('Training accuracy:', np.mean(train_acc), '+/-', np.std(train_acc))
    print('Testing accuracy:', np.mean(test_acc), '+/-', np.std(test_acc))
    print()
    print('Training macro-F1:', np.mean(train_f1), '+/-', np.std(train_f1))
    print('Testing macro-F1:', np.mean(test_f1), '+/-', np.std(test_f1))

    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

############################
# SUMMARY PROCESSING UTILS #
############################

def count_word_frequencies(summaries):
    """
    For an iterable of strings `summaries` returns a list
    of pairs (word, count) sorted by count descending.
    """
    word_counts = dict()
    for summary in summaries:
        for word in summary.split():
            count = word_counts.get(word, 0)
            word_counts[word] = count + 1
    return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

def count_ngram_frequencies(summaries, n):
    """
    ngram is any a substring of length n.
    For an iterable of strings `summaries` returns a list
    of pairs (ngram, count) sorted by count descending.
    """
    ngram_counts = dict()
    for summary in summaries:
        for i in range(0, len(summary) - n + 1):
            ngram = summary[i:(i+n)]
            count = ngram_counts.get(ngram, 0)
            ngram_counts[ngram] = count + 1
    return sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)

def summaries_to_vectors(summaries, words):
    """
    Converts a list of summaries to vectors using a bag of words technique.
    Only words in 'words' are considered.
    """
    vectors = np.zeros([len(summaries), len(words)])
    word_to_index = dict(((w, i) for i, w in enumerate(words)))
    for i, summary in enumerate(summaries):
        for word in summary.split():
            wi = word_to_index.get(word)
            if wi is not None:
                vectors[i, wi] += 1
    return vectors

def stem_summaries(summaries):
    """
    Returns stemmed summaries.
    To make stemmsk work rewrite unicode -> str, add import sys.
    """
    new_summaries = []
    for summary in summaries:
        new_summary = []
        for word in summary.split():
            new_summary.append(stem(word))
        new_summaries.append(' '.join(new_summary))
    return new_summaries

