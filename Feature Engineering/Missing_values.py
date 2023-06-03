#Kütüphaneleri import etme
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.preprocessing import MinMaxScaler

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

# EDA
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

#Missing values
df.isnull().values.any() #eksik gözlem var mı yok mu sorgusu
df.isnull().sum() #değişkenlerdeki eksik değer sayısı
df.isnull().sum().sum() #veri setindeki toplam eksik değer sayısı
df.notnull().sum() #değişkenlerdeki tam değer sayısı
df[df.isnull().any(axis=1)] #en az bir tane eksik değere sahip olan gözlem birimleri
df[df.notnull().all(axis=1)] #tam olan gözlem birimleri
df.isnull().sum().sort_values(ascending=False) #Azalan şekilde sıralama işlemi
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False) #yüzdelik olarak eksik değer oranı azalan
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0] #eksik değeri olan değişkenlerin adını verir

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)
missing_values_table(df, True)


dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
#ilgili değişkenin tipi object den farklı ise bu değişkeni ortalaması ile doldur, object ise olduğu gibi kalsın
dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
df["Embarked"].fillna("missing") #string ifade ile doldurma işlemi yapılailir analiz için

#kategorik değişkenlerde doldurma işlemi
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

# Kategorik Değişken Kırılımında Değer Atama
df.groupby("Sex")["Age"].mean()
#cinsiyet kırılımında yaş değişkeninin ortalamasını al eksiklikleri doldur.
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum() #fillna doldur
df.groupby("Sex")["Age"].mean()["female"] #kadın kırılımında ortalama

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]
df.isnull().sum()


#Tahmine Dayalı Atama İşlemleri
df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True) #Binary şekilde tutma
dff.head(10)

# değişkenlerin standartlatırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5) #komşuluk sayısı 5
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
df["age_imputed_knn"] = dff[["Age"]]
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]] #kontrol
df.loc[df["Age"].isnull()]


# Eksik Veri Yapısının İncelenmesi
msno.bar(df) #tam sayı olan gözlem sayılarını verir
plt.show()

msno.matrix(df) #değişkenlerdeki eksikliklerin birlikte çıkıp çıkmadığını anlamamızı sağlar
plt.show()

msno.heatmap(df) #Isı haritası
plt.show()


# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Survived", na_cols)


#TEKRAR

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)