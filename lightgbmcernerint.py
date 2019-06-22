import warnings 
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
import numpy as np 
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold as SKF
import gc
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import classification_report,auc,matthews_corrcoef
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals.joblib import parallel_backend
from lightgbm import LGBMClassifier
import lightgbm as lgb
inpath=['/repo4/khsieh/cernerimp/notpimpnu1095.csv','/repo4/khsieh/cernerimp/notpimpnu1460.csv','/repo4/khsieh/cernerimp/notpimpnu1825.csv']

classpath=['/repo4/khsieh/cernerimp/allcodenabf1095class.csv','/repo4/khsieh/cernerimp/allcodenabf1460class.csv','/repo4/khsieh/cernerimp/allcodenabf1825class.csv']

imppath=['/repo4/khsieh/cernerimp/result/notp_1095pimpnu.csv','/repo4/khsieh/cernerimp/result/notp_1460pimpnu.csv','/repo4/khsieh/cernerimp/result/notp_1825pimpnu.csv']

imppathn=['/repo4/khsieh/cernerimp/result/notp_1095pimpnnu.csv','/repo4/khsieh/cernerimp/result/notp_1460pimpnnu.csv','/repo4/khsieh/cernerimp/result/notp_1825pimpnnu.csv']
otherreportpath=['/repo4/khsieh/cernerimp/result/lrclfreportotnontpgbmnonu.csv']
def shuffle (train_y,train_weight):
    totalnu=len(train_y)
    totalindices=np.arange(totalnu)
    train_y=train_y[np.random.permutation(totalindices)]
    train_weight=train_weight[np.random.permutation(totalindices)]
    return train_y,train_weight
def gbmtrain (params,train_x,train_y,train_weight,val_x, val_y,val_weight,featurename):
    dtrain = lgb.Dataset(train_x,train_y,weight=train_weight,free_raw_data=True, silent=True)
    dval = lgb.Dataset(val_x, val_y,weight=val_weight,reference=dtrain,free_raw_data=True, silent=True)
    clf = lgb.train(params=params, train_set=dtrain,valid_sets=dval)
    imp_df = pd.DataFrame()
    imp_df['feature'] = featurename
    imp_df['importance_gain'] = clf.feature_importance(importance_type='gain')
    imp_df['importance_split'] = clf.feature_importance(importance_type='split')
    return imp_df, clf

def impdf_all_fu(imp_df_all,threshold):
    imp_df_all_normal=imp_df_all[['feature','importance_split']].groupby(['feature'],as_index=False).mean()
    imp_df_all_normal = imp_df_all_normal.sort_values('importance_split', ascending = False).reset_index(drop = True)
    imp_df_all_normal['normalized_importance'] = imp_df_all_normal['importance_split'] / imp_df_all_normal['importance_split'].sum()
    imp_df_all_normal['cumulative_importance'] = np.cumsum(imp_df_all_normal['normalized_importance'])
    imp_df_all_normal= imp_df_all_normal.sort_values('cumulative_importance')
    record_low_importance = imp_df_all_normal[imp_df_all_normal['cumulative_importance'] > threshold]
    return  imp_df_all_normal,record_low_importance
skf=SKF(n_splits=500,random_state=3,shuffle=True)
lightpara={'objective':'binary','n_estimators':1000,'learning_rate':0.05,'num_leaves':50,'tree_learner':'data','num_threads':16,'bagging_fraction':0.8,'feature_fraction':0.8,'metric':'auc'}
clfreportall=pd.DataFrame()
otherreportall=pd.DataFrame()
inpathlb=['365lgbnu','730lgbnu','1095lgbnu','1460lgbnu','1825lgbnu']
aucresult1=[]
accresult1=[]
mccresult1=[]
for k in range(len(inpath)):
    print("start reading")
    print (inpath[k])
    table=pd.read_csv(inpath[k])
    table=table[['Uid','variable','status']]
    table=table.drop_duplicates()
    patientclass=pd.read_csv(classpath[k])
    patientclass=patientclass[['Uid','class','t2dmclass','controlclass','classweight']]
    totaluid=patientclass['Uid'].values
    totalnu=len(totaluid)
    totalindices=np.arange(totalnu)
    totalgroup=patientclass['class'].values
    try3=table['variable'].drop_duplicates().reset_index()
    try3=try3['variable'].to_frame()
    all_imp=pd.DataFrame()
    all_imp_n=pd.DataFrame()
    all_imp_low=pd.DataFrame()
    for train_val_index, test_index in tqdm (skf.split(totalindices,y=totalgroup,groups=totalgroup)):
        aucresult=[]
        resultday=[]
        print ('strart build model')
        batchuid=totaluid[test_index]
        newpatientclass=patientclass[patientclass['Uid'].isin(batchuid)]
        newpatientdata=table[table['Uid'].isin(batchuid)]
        try61=newpatientdata[newpatientdata['Uid']==batchuid[1]]
        try61=try61[['Uid','variable','status']]
        try61=pd.merge(try61,try3,how='outer')
        filllist={'Uid':batchuid[1],'status':0}
        try61=try61.fillna(value=filllist)
        newpatientdata=pd.concat([newpatientdata,try61])
        newpatientdata=newpatientdata.pivot_table(index='Uid',values='status',columns='variable').fillna(0)
        newpatientdata=pd.merge(newpatientdata,newpatientclass,on='Uid')
        newpatientclass=newpatientdata['class']
        newpatientclass=newpatientclass.values
        newpatientweight=newpatientdata['classweight']
        newpatientweight=newpatientweight.values
        newpatientdata=newpatientdata.drop(columns=['Uid','class','t2dmclass','controlclass','classweight'])
        feature=newpatientdata.columns
        newpatientdata1=newpatientdata.values
        del newpatientdata
        gc.collect()
        traintestnu=newpatientdata1.shape[0]
        traintestindice = np.arange(traintestnu)
        trainindice,testindice=train_test_split(traintestindice,test_size=0.2,stratify=newpatientclass,random_state=3)
        traindata=newpatientdata1[trainindice]
        testdata=newpatientdata1[testindice]
        trainlabel=newpatientclass[trainindice]
        testlabel=newpatientclass[testindice]
        trainweight=newpatientweight[trainindice]
        testweight=newpatientweight[testindice]
        print ('model start')
        imp_n, clf=gbmtrain(lightpara,traindata,trainlabel,trainweight,testdata, testlabel,testweight,feature)
        all_imp_n=pd.concat([all_imp_n,imp_n])
        clfresult=clf.predict(testdata)
        
        
        auc=roc_auc_score(testlabel,clfresult)
 
        otherreport=pd.DataFrame()
        aucresult.append(auc)
        
        resultday.append(inpathlb[k])
        otherreport['auc']=aucresult
        
        otherreport['data']=resultday
        otherreportall=pd.concat([otherreportall,otherreport], axis=0)
        otherreportall.to_csv(otherreportpath[0])
        print(inpathlb[k],auc)
        
        for i in range(10):
            new_trainlabel,new_trainweight=shuffle (trainlabel,trainweight)
            new_testlabel,new_testweight=shuffle (testlabel,testweight)
            imp, _=gbmtrain(lightpara,traindata,new_trainlabel,new_trainweight,testdata, new_testlabel,new_testweight,feature)
            all_imp=pd.concat([all_imp,imp])
    all_imp_n.to_csv(imppathn[k])
    all_imp.to_csv(imppath[k])