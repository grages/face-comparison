from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import math


def initgnb():
    clf = joblib.load('./data/bay_vgg_1.pkl')
    return clf


def calcos(a,b):
    tmp1=0.0
    tmp2=0.0
    tmp3=0.0
    for i in range(0,len(a)):
        tmp1+=a[i]*b[i]
        tmp2+=a[i]*a[i]
        tmp3+=b[i]*b[i]
    tmp2=math.sqrt(tmp2)
    tmp3=math.sqrt(tmp3)
    result=tmp1/tmp2/tmp3
    return result

def gnb(x,cla):
    re=cla.predict([x])
    r=re[0]
    return r

