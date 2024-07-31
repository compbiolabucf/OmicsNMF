import numpy as np, torch, matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from  matplotlib.colors import LinearSegmentedColormap
custom_cmap=LinearSegmentedColormap.from_list('rg',["g", "black", "r"], N=256) 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib.pylab as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import numpy as np 
import pandas as pd 
from scipy.stats import ttest_ind


def get_classifier():
    return RandomForestClassifier(n_estimators=50, random_state=42)

def nmf_U_V(X,k=10,init='random', random_state=1111):
    nmf_model = NMF(n_components=k, init=init, random_state=random_state)
    U = nmf_model.fit_transform(X)
    V = nmf_model.components_
    return U,V

def get_prediction_auc(train_loader, valid_loader, test_loader,gen,label_path,device = 'cpu'):
    if(label_path is not None):
        label = pd.read_csv(label_path)
        label.rename(columns={x:y for x,y in zip(['Unnamed: 0', 'ER', 'TN'],["sample","TN","ER"])},inplace=True)
        label = label.set_index("sample")
        label = label.T
    else:
        label = None
    plt.switch_backend('agg')
    plt.tight_layout()
    all_source = []
    all_samples = []
    er_label = []
    xs_train = []
    xs_valid = []
    xs_test = []
    mirna_true_train = []
    y_test = []
    y_train = []
    y_valid = []
    sample_size = train_loader.dataset.__len__() + valid_loader.dataset.__len__()
    for i in range(train_loader.dataset.__len__()):
        sample = train_loader.dataset.train_samples[i]
        x,_,xs,y,_ = train_loader.dataset.__getitem__(i)
        all_source.append(x)
        all_samples.append(sample)
        if(label is not None):
            if(sample in label):
                er = label[sample]['ER']
                mirna_true_train.append(y)
                xs_train.append(x)
                y_train.append(er)
    print(len(mirna_true_train))
    for i in range(valid_loader.dataset.__len__()):
        sample = valid_loader.dataset.train_samples[i]
        x,_,xs,y,_ = valid_loader.dataset.__getitem__(i)
        all_source.append(x)
        all_samples.append(sample)
        if(label is not None):
            if(sample in label):
                er = label[sample]['ER']
                mirna_true_train.append(y)
                xs_valid.append(x)
                y_valid.append(er)
    for i in range(test_loader.dataset.__len__()):
        sample = test_loader.dataset.train_samples[i]
        x,_,xs,y,_ = test_loader.dataset.__getitem__(i)
        all_source.append(x)
        all_samples.append(sample)
        if(label is not None):
            if(sample in label):
                er = label[sample]['ER']
                mirna_true_train.append(y)
                xs_test.append(x)
                y_test.append(er)
    all_source = torch.tensor(np.array(all_source))
    xs_train = torch.tensor(np.array(xs_train))
    y_train = np.array(y_train)
    xs_valid = torch.tensor(np.array(xs_valid))
    y_valid = np.array(y_valid)
    xs_test = torch.tensor(np.array(xs_test))
    y_test = np.array(y_test)
    # mirna_true_train = np.array(mirna_true_train)
    gen.eval()
    output = gen(all_source.to(device))
    all_target = output.detach().cpu().numpy()
    target_df = pd.DataFrame(data=all_target.T, columns = all_samples)
    if(label is None):
        return -1, -1, -1, None, target_df
    
    output = gen(xs_train.to(device))
    miRNA_gen_train = output.detach().cpu().numpy()
    output = gen(xs_valid.to(device))
    miRNA_gen_valid = output.detach().cpu().numpy()
    variance_gen = np.var(miRNA_gen_train)
    output = gen(xs_test.to(device))
    miRNA_gen_test = output.detach().cpu().numpy()

    cv = StratifiedKFold(n_splits=5,random_state=1111, shuffle=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 1
    X = np.concatenate((miRNA_gen_train,miRNA_gen_valid,miRNA_gen_test),axis=0)
    Y = np.concatenate((y_train,y_valid,y_test))
    fig, ax = plt.subplots(1,1)
    for (train_idx, test_idx) in cv.split(X, Y):
        X_train = X[train_idx]
        X_test = X[test_idx]
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]
        rfc = get_classifier()
        rfc.fit(X_train, Y_train)
        prediction = rfc.predict_proba(X_test)
        fpr, tpr, t = roc_curve(Y_test, prediction[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=2, alpha=0.5, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i= i+1


    # plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color='blue',
            label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    fig.suptitle('ROC 5 fold cv on available+missing data')
    ax.legend(loc="lower right")

    #Trained on available data, tested on missing data
    mean_fpr = np.linspace(0,1,100)
    rfc = RandomForestClassifier(n_estimators=10, random_state=1111)
    rfc.fit(np.concatenate((miRNA_gen_train,miRNA_gen_valid),axis=0), np.concatenate((y_train,y_valid)))
    prediction = rfc.predict_proba(miRNA_gen_test)
    fpr, tpr, t = roc_curve(y_test, prediction[:, 1])
    roc_auc_1 = auc(fpr, tpr)

    #Trained on training + missing data tested on validation data
    rfc = RandomForestClassifier(n_estimators=10, random_state=1111)
    rfc.fit(np.concatenate((miRNA_gen_train,miRNA_gen_test),axis=0), np.concatenate((y_train,y_test)))
    prediction = rfc.predict_proba(miRNA_gen_valid)
    fpr, tpr, t = roc_curve(y_valid, prediction[:, 1])
    roc_auc_2 = auc(fpr, tpr)
    return mean_auc, roc_auc_1, roc_auc_2, fig, target_df