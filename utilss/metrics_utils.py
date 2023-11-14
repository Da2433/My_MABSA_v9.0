import numpy as np

from sklearn.metrics import precision_recall_fscore_support



def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    #f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro