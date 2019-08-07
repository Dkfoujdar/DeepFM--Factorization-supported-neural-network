import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import label_ranking_average_precision_score, make_scorer #accuracy_score, roc_auc_score,
from sklearn.metrics import log_loss
from Metrix import gini_norm
from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit

import config
from DataReader import FeatureDictionary, DataParser
sys.path.append("..")
from DeepFM import DeepFM

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)



def _load_data(): #we just loaded the data and then processed it

    dfTest = pd.read_csv("C:/Users/Amit/Desktop/FMNN/data/test.csv")
    dfTrain = pd.read_csv("C:/Users/Amit/Desktop/FMNN/data/train.csv")
	
    cols = [c for c in dfTrain.columns if c not in ["id","ImTyIT1","ImTyIT2","ImTyIT4","ImTyIT5"]]
    X_train = dfTrain[cols].values
    ycol = [t for t in dfTrain.columns if t in ["ImTyIT1","ImTyIT2","ImTyIT4","ImTyIT5"]]
    y_train = dfTrain[ycol].values #target or the output values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values
    

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest)
    data_parser = DataParser(feat_dict=fd)        #converting into-Xi: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)
    
    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0]) 

    y_train_meta = np.zeros((dfTrain.shape[0], 4), dtype=float) #crated array of (rows in dfTrain,4)
    y_test_meta = np.zeros((dfTest.shape[0], 4), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0:4] += dfm.predict(Xi_test, Xv_test)
        b = np.zeros_like(y_train_meta)
        b[np.arange(len(y_train_meta)), y_train_meta.argmax(1)] = 1
        #y_train_meta = np.array(y_train_meta, dtype=np.float32)
        gini_results_cv[i] = label_ranking_average_precision_score(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))
    #b = np.zeros_like(y_test_meta)
    #b[np.arange(len(y_test_meta)), y_test_meta.argmax(1)] = 1
    #y_test_meta = b

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    #filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _final_result(ids_test, y_test_meta, filename="result.csv")

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta

def _final_result(ids, y_pred, filename="result.csv"):
    pd.DataFrame({"id": ids, "ImTyIT1": y_pred[:,0],"ImTyIT2": y_pred[:,1],"ImTyIT4": y_pred[:,2],"ImTyIT5": y_pred[:,3]}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")

def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("C:/Users/Amit/Desktop/FMNN/%s.png"%model_name)
    plt.close()


# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test = _load_data()

# folds
kf = ShuffleSplit(n_splits=3,train_size=0.8, test_size=0.2, random_state=69)
kf.get_n_splits(X_train,y_train)

print(kf)  

folds = list(kf.split(X_train,y_train))
   #print("TRAIN:", train_index, "TEST:", test_index)
   #X1, X = X_train[train_index], X_train[test_index]
   #y1, y = y_train[train_index], y_train[test_index]



#folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
#                             random_state=config.RANDOM_SEED).split(X_train, y_train))


# ------------------ DeepFM Model ------------------
# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 20,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm,
    "random_seed": config.RANDOM_SEED
}
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)


# ------------------ FM Model ------------------
#fm_params = dfm_params.copy()
#fm_params["use_deep"] = False
#y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)


# ------------------ DNN Model ------------------
#dnn_params = dfm_params.copy()
#dnn_params["use_fm"] = False
#y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)

