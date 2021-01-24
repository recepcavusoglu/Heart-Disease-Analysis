import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#1- Veri setinde hastalıklı ve sağlam sayılarını sütun grafiği kullanarak çizdirin.
def Hasta_Dagilim(p_df):
    status=[0,0]
    status[0]=len(p_df[p_df["sex"]==1])
    status[1]=len(p_df)-status[0]
    labels=['Hasta','Hasta Değil']
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, status, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.title('Hastalık Dağılımı')
    plt.show()

#2- Cinsiyete göre hasta ve sağlıklı hasta sayıları sütun grafiği ile ifade ediniz. 
def Hasta_Cinsiyet_Dagilim(p_df):
    male_count=len(p_df[p_df["sex"]==1])
    sick_male_count=len(p_df[(p_df["sex"]==1)&(p_df["target"]==0)])
    female_count=len(p_df[p_df["sex"]==0])
    sick_female_count=len(p_df[(p_df["sex"]==0)&(p_df["target"]==0)])
    #sick_male,healty_male,sick_female,healty_female
    status=[sick_male_count,male_count-sick_male_count,sick_female_count,female_count-sick_female_count]
    labels=['Erkek Hasta','Erkek Sağlıklı','Kadın Hasta','Kadın Sağlıklı']
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, status, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.title('Hastalık Cinsiyet Dağılımı')
    plt.show()

#3- Veri setindekilerin yaş dağılımını gösteren bir sütun grafiği çizdirin.
#4- Veri setinde hasta olanların yaş dağılımını gösteren bir sütun grafiği çizdirin.
#İkisini tek methotda birleştirdim parametre olarak True gelirse 4, False gelirse 3 durumunu inceler.
def Yas_Dagilimi(p_df,p_sick_status):
    plt_title="Veri Seti Yaş Dağılımı"
    if p_sick_status==True:
        p_df=p_df[p_df["target"]==0]
        plt_title="Hasta Yaş Dağılımı"
    age_distribution=[]
    age_distribution.append(len(p_df[p_df["age"]<30]))
    age_distribution.append(len(p_df[(p_df["age"]>=30)&(p_df["age"]<40)]))
    age_distribution.append(len(p_df[(p_df["age"]>=40)&(p_df["age"]<50)]))
    age_distribution.append(len(p_df[(p_df["age"]>=50)&(p_df["age"]<60)]))
    age_distribution.append(len(p_df[(p_df["age"]>=60)&(p_df["age"]<70)]))
    age_distribution.append(len(p_df[p_df["age"]>70]))
    labels=["0-29","30-39","40-49","50-59","60-69","70+"]
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, age_distribution, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.title(plt_title)
    plt.show()

#5- Veri setini sınıflandırma işlemi için hazırlayınız(Özelliklerin ölçeklenmesi, kategorik veri var ise nümerik hale getirilmesi, özelliklerin ve çıktının değişkenlere aktarılması)
#6- Veri setini sınıflandırma algoritmalarında kullanmak üzere eğitim ve test olmak üzere ayırın. (%80 eğitim ,%20 test olacak şekilde)
def Data_Preprocess(p_df):
    min_max_scaler=preprocessing.MinMaxScaler()
    data_scaled=min_max_scaler.fit_transform(p_df.values)
    df=pd.DataFrame(data_scaled,columns=p_df.columns)
    labels=df.target.copy()
    features=df.drop(["target"],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=1702)
    return X_train, X_test, y_train, y_test

if __name__=="__main__":
    df=pd.read_csv('data.csv')
    X_train, X_test, y_train, y_test=Data_Preprocess(df)