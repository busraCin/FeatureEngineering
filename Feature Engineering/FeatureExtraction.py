
import pandas as pd

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
# Feature Extraction (Özellik Çıkarımı), ham veriden değişken üretmek.

# Binary Features: Flag, Bool, True-False

#Cabin - Survived
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int') #dolu olanlara 1 boş olanlara (NaN) 0
df.groupby("NEW_CABIN_BOOL").agg({"Survived" : "mean"})

#istatistik Oran Testi 1
from statsmodels.stats.proportion import proportions_ztest #count başarı sayısı , nobs gözlem sayısı
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(), #kabin numarası olan ve hayatta kalan kaç kişi var
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0], #kabin numarası olanlar kaç kişi
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #Çok değişkenli etkiyi bilmiyoruz.

#Parch, SibSp - Survived
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO" #Gemide yanlız değilmiş
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES" #Yanlız
df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

#istatistik Oran Testi 2
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
df.head()

# Text'ler Üzerinden Özellik Türetmek
df["NEW_NAME_COUNT"] = df["Name"].str.len() #Letter Count ,harf
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" "))) # Word Count, Kelime
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))# Özel Yapıları Yakalamak, Dr
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})

# Regex ile Değişken Türetmek
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False) #boşluk, büyük harf, küçük, harf nokta
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

# Feature Interactions (Özellik Etkileşimleri)
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1 #aile boyutu
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'
df.head()
df.groupby("NEW_SEX_CAT")["Survived"].mean() #hayatta kalma ornları