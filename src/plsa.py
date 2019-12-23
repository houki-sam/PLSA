#-*-encoding:utf-8-*-
import os
from time import time

import numpy as np
import pandas as pd

from .one_hot import one_hot
from .one_hot import product


class PLSA(object):
    def __init__(self, multi, Z):
        #読み込みデータ
        self.multi = multi #多項分布
    
        #潜在クラス数
        self.Z = Z 

        #one_hot表現(self.on)とlabel(self.label)を作成
        self.one_hot, self.label = one_hot(self.multi) 

        #------初期化--------------------------
        #P(z)
        self.Pz = np.random.rand(self.Z)
        self.Pz /=np.sum(self.Pz) 
        
        #多項分布の確率分布P_multi_z
        self.P_multi_z = [] #初期化

        for x in self.label: #配列に要素を追加
            stack = np.random.rand(len(x), self.Z)
            stack /= np.sum(stack , axis=0)[None,:]#正規化
            self.P_multi_z.append(stack)
    

    #============E-STEP=========================
    def estep(self):
        #多項分布の行列計算
        tt=[self.one_hot[x]@self.P_multi_z[x] for x in range(len(self.label))]
        tmp_multi = product(tt)
        

        #E-STEPの値はself.tmpで表現する
        self.tmp = self.Pz[None,:] * tmp_multi
        self.tmp /= np.sum(self.tmp,axis=1)[:,None]
        print(self.tmp.shape)

        #nan or infが出たら置換
        self.tmp[np.isnan(self.tmp)] = 1/self.Z
        self.tmp[np.isinf(self.tmp)] = 1/self.Z

    #=========M-STEP===============================================
    def mstep(self):
        #---------P(z)---------------------------
        self.Pz = np.sum(self.tmp, axis = 0)
        self.Pz /= np.sum(self.Pz)

        
        #-------多項分布------------------
        for x in range(len(self.label)):
            stack= np.dot(self.one_hot[x].T, self.tmp)
            stack /= np.sum(stack, axis=0)[np.newaxis,:]
            self.P_multi_z[x] = stack


    #=======学習フェーズ======================================
    def train(self, k=1000, t=1.0e-7):
        # 対数尤度が収束するまでEステップとMステップを繰り返す
        prev_llh = np.inf #対数尤度の初期値
        # flagを作成
        flag = False

        '''
        ステップ数分だけ実行
        収束したら実行をストップ
        '''
        for i in range(k):
            self.estep()
            self.mstep()
            llh = self.llh()
            if abs((llh - prev_llh) / prev_llh) < t:
                flag = True
                self.write_result()
                break
            
            prev_llh = llh
        if flag:
            print("学習終了。\n"+str(i)+"回で終了しました。")
        else:
            print("収束しませんでした。")

    #=======対数尤度===========================================
    def llh(self):
        #多項分布の計算
        tmp_multi = [self.one_hot[x]@self.P_multi_z[x] for x in range(len(self.label))]
        tmp_multi = product(tmp_multi)

        #対数尤度を作成
        log = np.log(np.sum(self.Pz[None,:] * tmp_multi, axis=1))

        #不定値が出た場合、置換する。
        log[np.isnan(log)] = -100
        log[np.isinf(log)] = -100

        log = np.sum(log)
        return log


    #======結果書き込み=================================================
    def write_result(self):
        #resultファイルがなければ作成
        if not os.path.exists("result"):
            os.makedirs("result")
        #潜在クラス数ごとにファイルを作成
        if not os.path.exists("result/"+str(self.Z)):
            os.makedirs("result/"+str(self.Z))
        
        #潜在クラスの所属確率を記録
        np.savetxt("result/"+str(self.Z)+"/Pz.csv",self.Pz,delimiter=",")
        
        
        for x in range(len(self.label)):
            header=""
            for name in self.label[x]:
                header += str(name)+","
            np.savetxt("result/"+str(self.Z)+"/multi_"+str(x)+".csv",self.P_multi_z[x].T\
                ,delimiter=",", header=header[:-1],comments='')