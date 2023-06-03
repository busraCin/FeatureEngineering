#Kütüphaneleri import etme
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

#Görsel ayarlamalar
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#veri okuma işlemleri

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data
df = load()
df.head()

#IQR HESABI İLE AYKIRI DEĞER YAKALAMA
def outlier_thresholds(dataframe, col_name, q1=0.25 , q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age") #yaş değişkeni için eşik değerlere bakalım

#Aykırı değer var mı yok mu öğrnelim, bool
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")#yaş değişkeni için bakalım

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car kategorik değişken analizi
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]  #O = object
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols sayısal değişken analizi
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}") #gözlem
    print(f"Variables: {dataframe.shape[1]}") #değişken = aşağıdaki 3ünün toplamı.
    print(f'cat_cols: {len(cat_cols)}') #kategorik değişen
    print(f'num_cols: {len(num_cols)}') #sayısal değişken
    print(f'cat_but_car: {len(cat_but_car)}') #kategorik ama kardinal
    print(f'num_but_cat: {len(num_but_cat)}') #sayısal ama kategorik değişken,  raporlama için var
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df) #titanik veri seti için analiz edeliim.
num_cols = [col for col in num_cols if col not in "PassengerId"] #istisna çıkartma

#aykırı değerler var mı
for col in num_cols:
    print(col, check_outlier(df, col))


# Aykırı Değerlerin Kendilerine Erişmek
def grab_outliers (dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe,col_name)

    if dataframe[( (dataframe[col_name] < low) | (dataframe[col_name] > up) )].shape[0] > 10:
        print( dataframe[( (dataframe[col_name] < low) | (dataframe[col_name] > up) )].head() )

    else: print (dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[( (dataframe[col_name] < low) | (dataframe[col_name] > up) )].index
        return outlier_index


grab_outliers(df, "Age")
grab_outliers(df, "Age", True)
age_index = grab_outliers(df, "Age", True)


#Yakalanan Aykırı Değerleri Veri Setinden Silme
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))] #aykırı olmayan değerleri al
    return df_without_outliers


for col in num_cols:
    new_df = remove_outlier(df, col) #tüm veri seti için

df.shape[0] - new_df.shape[0]   #silinen aykırı değer sayısı

"""Baskılama Yöntemi(re-assignment with thresholds).
 Bir hücredeki aykırılıktan dolayı diğer hücrelerdeki verilerden olmamak için aykırı değer baskınlanır.
 Eşik değerin üstünde veya altında olan değerler eşik değeri ile değişir."""
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col)) #aykırı değer var mı
for col in num_cols:
    replace_with_thresholds(df, col) #baskılama yöntemi
for col in num_cols:
    print(col, check_outlier(df, col)) #tekrar bak aykırı değer var mı


#Local Outlier Factor (LOF)
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64']) #sadece sayısal değerler
df = df.dropna() #eksik değerleri düşür
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_thresholds(df, "carat")
df[((df["carat"] < low) | (df["carat"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20) #komşuluk sayısı 20
clf.fit_predict(df) #skorları hesapla

df_scores = clf.negative_outlier_factor_ #hesaplanan skorları tut
df_scores[0:5] #ör 5 tanesini gözlemliyelim
""" df_scores = -df_scores ile skorlar pozitif tutulur. Eşik değerleri belirlerken daha rahat okunabilirlik açısından 
 elbow yönetmiyle negatif değerler tercih edlir."""
np.sort(df_scores)[0:5] #en kötü skora sahip 5 değer

#Elbow yöntemi, genelde grafikteki en dik noktadan seçilir
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[3] #sıralamada ki 3. değeri eşik değer olarak belirle
df[df_scores < th] #outliers
df[df_scores < th].shape
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T #neden aykırı olduğunu anlamak için

df[df_scores < th].index
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

























