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

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data
df = load_application_train()
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

# Label Encoding & Binary Encoding

df["Sex"].head()
le = LabelEncoder()
le.fit_transform(df["Sex"])[0:13] #label encoding işlemini yap alfabetik, ilk 13 değeri getir
le.inverse_transform([0 , 1]) #0 ve 1 değerleri hangi sınıf değerine denk geliyor

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
df = load()

#iki sınıflı kategorik değişkenleri seçme ve label encoderden geçirme
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", float, int]
               and df[col].nunique() == 2] #nunique eksik değeri sınıf olarak görmez

for col in binary_cols:
    label_encoder(df, col)
df.head()

#daha büyük bir veri setinde label encoding işlemi yapalım
df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float, "int64", "float64"]
               and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)
#eksik değerleri 2 ile doldurmuş

df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique()) #eksik değeri (nan) de eşsiz saydığını anlıyoruz.


# One-Hot Encoding

df = load()
df.head()
df["Embarked"].value_counts() #sınıfları arasında ölçeklenebilir bir fark yok.

pd.get_dummies(df, columns=["Embarked"]).head() #verilen ilgili değişkeni dönüştürür
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head() #ilk sınıf drop edilir, alfabetik seçilir.
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head() #eksik değerler için de ayrı bir sınıf oluşturma işlemi
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head() #hem label encoding işlemi hem one-hot encoding yapıldı

#fonksiyonlaştırma
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df) #elimizdeki kategorik değişkenlere bakalım
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2] # kategorik değişkenlerde eşsiz değer sayısı 2 10 arası olan sınıfları al
#ohe_cols değerlerine one-hot encoding uygula
one_hot_encoder(df, ohe_cols).head()


# Rare Encoding
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts() #frekans değerlerine bakalım
cat_cols, num_cols, cat_but_car = grab_col_names(df)
#onlarca kategorik değişkenin onlarca sınıfı var. karmaşıklığı azaltmak için eşik değeri belirlenip rare encoding işlemi uyguluyalım
def cat_summary(dataframe, col_name, plot=False): #kategorik değişkenleri detaylı analiz edelim
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})) #oran, yüzdelikler bütün veriye göre hesaplanır.
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
for col in cat_cols:
    cat_summary(df, col)

#2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean() #bağımlı değişken açısından inceleme
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts())) #ilgili categorik değişkenin kaç sınıfı var
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),  #sınıf frekansları
                            "RATIO": dataframe[col].value_counts() / len(dataframe), #sınıf oranları
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n") #bağımlı değişkene göre groupby işlemi
rare_analyser(df, "TARGET", cat_cols) #Rare analizi

# 3. Rare encoder'ın yazılması
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy() #girilen dataframe in kopyasını al üzerinde değişiklik yapılacağı için
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O' #kategorik değişken ise ve
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)] #(sınıf frekansları / toplam gözlem sayısı)nrare_perc değerinden herhangi bir tanesi küçük ise
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df) #sınıf oranlarını hesapla
        rare_labels = tmp[tmp < rare_perc].index #eşik değerden düşük oraanda sınıf değeri olan değerleri tur
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var]) #rare colums içinde bir tane bile rare labels varsa, rareları gördüğün yerlere Rare yaz
    return temp_df

new_df = rare_encoder(df, 0.01)
rare_analyser(new_df, "TARGET", cat_cols)

# Feature Scaling (Özellik Ölçeklendirme)
#eşit şartlarda yaklaşım,
#eğitim süresini hızlandırmak ve
#uzaklık temelli yöntemlerde yanlılığın önüne geçmek için özellik ölçeklendirme yapılır.

# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

# RobustScaler: Medyanı çıkar iqr'a böl. Aykırı değerlere daha dayanıklı
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T
df.head()

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False): #sayısal değişkeni çeyreklik değeri gösterir ve grafiğini oluşturur (histogram)
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot: #yapılarını kordu mu kontrol edelim
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True) #block true ile grafikteki hatayı gider.

for col in age_cols:
    num_summary(df, col, plot=True)


# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning

df["Age_qcut"] = pd.qcut(df['Age'], 5) #hangi değişken, kaç parca olacağını ver (5)