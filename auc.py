# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:18:39 2018

@author: 11955
"""

import numpy as np
import pandas as pd

from data_preprocess import *
from offline_train_label import *
from coupon_feature import *
from time_feature import *
from offline_features import *
from online_features import *
from get_train_test import *

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def predict(offline_train,online_train):
    print("数据预处理中......")
    offline_train = trainDataPreprocess(offline_train)
    online_train = trainDataPreprocess(online_train,True)
    
    print("打标签")
    offline_train = getOfflineLabel(offline_train)
    
    print("特征工程")
    offline_train = getCouponFeatures(offline_train)
    offline_train = getTimeFeatures(offline_train)
    
    print("offline_features")
    #offline_features
    X_U_off = offline_train[["User_id"]]
    X_M_off = offline_train[["Merchant_id"]]
    X_UM_off = offline_train[["User_id","Merchant_id"]]
    
    X_U_off = X_U_off.drop_duplicates() ###***X_U = X_U.drop_duplicates()是有返回值的  merge时注意要去重
    X_M_off = X_M_off.drop_duplicates()
    X_UM_off = X_UM_off.drop_duplicates()
    
    X_off_user = getOfflineUserFeatures(offline_train,X_U_off)
    X_off_mer = getOfflineMerchantFeatures(offline_train,X_M_off)
    
    print("online_features")
    #online_features
    X_U_on = online_train[["User_id"]]
    X_M_on = online_train[["Merchant_id"]]
    X_UM_on = online_train[["User_id","Merchant_id"]]
    
    X_U_on = X_U_on.drop_duplicates()
    X_M_on = X_M_on.drop_duplicates()
    X_UM_on = X_UM_on.drop_duplicates()
    
    X_on_user = getOnlineUserFetures(online_train,X_U_on)
    
    print("get train")
    #get train test
    #merge
    train = pd.merge(offline_train,X_off_user,on="User_id",how="left")
    train = pd.merge(train,X_off_mer,on="Merchant_id",how="left")
    train = pd.merge(train,X_on_user,on="User_id",how="left")
    
    #X_train,y_train = getTrain(offline_train,X_off_user,X_off_mer,X_on_user)
    
    #切分tra,val
    tra = train[train["Date_received"]<=""]
    val = train[train["Date_received"]>""]
    
    y_tra = tra["label"]
    X_tra = train.drop(["User_id","Merchant_id","Coupon_id","Discount_rate","Date_received","Date","label"],axis=1)
    
    y_val = val["label"]
    X_val = val.drop(["User_id","Merchant_id","Coupon_id","Discount_rate","Date_received","Date","label"],axis=1)
    
    #训练数据集
    
    
    
    
    return train

if __name__ == "__main__":
    offline_train = pd.read_csv("data/ccf_offline_stage1_train.csv")
    online_train = pd.read_csv("data/ccf_online_stage1_train.csv")
    
    train = predict(offline_train,online_train)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    