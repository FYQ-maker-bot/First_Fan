import tensorflow as tf
import numpy as np
from sklearn import metrics
from keras.metrics import AUC
from sklearn.preprocessing import normalize
from model import getMatrixLabelh

new_model = tf.keras.models.load_model("model.h5")

train_file_name = r""
win1 = 100
X_test, T, raw, length, y_ture1 = getMatrixLabelh(train_file_name, win1)
num = 1450
X_test = X_test.reshape((num, 2000))

bbb = np.load(r"")
X2_test = np.zeros((num, 1547))
X2_test[:]=bbb[:]
X2_test = normalize(X2_test, axis=1, norm="l2")

y_pred = new_model.predict([X_test, X2_test])
y_class_1_prob = y_pred[:, 1]
y_test_pre = np.argmax(y_pred, axis=1)
fpr, tpr, thresholds1 = metrics.roc_curve(y_ture1, y_test_pre)
acc = metrics.accuracy_score(y_ture1, y_test_pre)
f1 = metrics.f1_score(y_ture1, y_test_pre)
precision = metrics.precision_score(y_ture1, y_test_pre)
recall = metrics.recall_score(y_ture1, y_test_pre)
auc = AUC()(y_ture1, y_class_1_prob)
mcc = metrics.matthews_corrcoef(y_ture1, y_test_pre)

print(':acc {:.4f}'.format(acc), ', mcc {:.4f}'.format(mcc), ',AUC {:.4f}'.format(auc),
      ',precision {:.4f}'.format(precision),',recall {:.4f}'.format(recall),
      ', f1 {:.4f}'.format(f1), ', sp {:.4f}'.format(1-fpr[1]))
