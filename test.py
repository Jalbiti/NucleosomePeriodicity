# -*- coding: utf-8 -*-
from nfr_clf import *
from sklearn import metrics

num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

window = 350
chr_feats = Features(num, window)
model350 = Model(chr_feats)
model350.predictor()
# model350.plot_learning(model350.history)
print(model350.score)

window = 250
chr_feats = Features(num, window)
model250 = Model(chr_feats)
model250.predictor()
# model250.plot_learning(model250.history)
print(model250.score)

window = 600
chr_feats = Features(num, window)
model600 = Model(chr_feats)
model600.predictor()
# model600.plot_learning(model600.history)
print(model600.score)

def plot_roc_curve(true_y, y_prob, name):
    """
    plots the roc curve based of the probabilities
    """
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr, label=name)
    plt.legend(name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


curr_model = model350
y_pred = curr_model.model.predict(curr_model.X_test)
plot_roc_curve(curr_model.y_test, y_pred, "350")

curr_model = model250
y_pred = curr_model.model.predict(curr_model.X_test)
plot_roc_curve(curr_model.y_test, y_pred, "250")

curr_model = model600
y_pred = curr_model.model.predict(curr_model.X_test)
plot_roc_curve(curr_model.y_test, y_pred, "600")

plt.title("Receiver Operating Characteristic (ROC) curve")
## plt.figure(figsize=(10,6))
plt.legend()
plt.savefig("ROC_w")

