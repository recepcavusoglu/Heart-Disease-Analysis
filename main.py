import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#1- Veri setinde hastalıklı ve sağlam sayılarını sütun grafiği kullanarak çizdirin.
def Hasta_Dagilim(p_df):
    status=[0,0]
    status[0]=len(p_df[p_df["target"]==1])
    status[1]=len(p_df)-status[0]
    labels=['Hasta','Hasta Değil']
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, status, align='center', alpha=0.5,color=['red','blue'])
    plt.xticks(y_pos, labels)
    plt.title('Hastalık Dağılımı')
    plt.show()

#2- Cinsiyete göre hasta ve sağlıklı hasta sayıları sütun grafiği ile ifade ediniz. 
def Hasta_Cinsiyet_Dagilim(p_df):
    
    male_count=len(p_df[p_df["sex"]==1])
    sick_male_count=len(p_df[(p_df["sex"]==1)&(p_df["target"]==1)])
    female_count=len(p_df[p_df["sex"]==0])
    sick_female_count=len(p_df[(p_df["sex"]==0)&(p_df["target"]==1)])
    #sick_male,healty_male,sick_female,healty_female
    status=[sick_male_count,male_count-sick_male_count,sick_female_count,female_count-sick_female_count]
    labels=['Erkek Hasta','Erkek Sağlıklı','Kadın Hasta','Kadın Sağlıklı']
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, status, align='center', alpha=0.5,color=['red','blue','red','blue'])
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
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=256)
    return X_train, X_test, y_train, y_test

#7- Logistic Regresyon kullanarak veri setini sınıflandırın
#(Sonucun karmaşıklık matrisini çizdirin, bu matrise göre Accuracy, Sensitivity, Specificity, Recall, Precision değerlendirme kriterlerini hesaplayın.) 
def Logistic(p_trainX,p_testX,p_trainY,p_testY):
    model=LogisticRegression()
    model.fit(p_trainX,p_trainY)
    y_pred = pd.Series(model.predict(p_testX))

    cnf_matrix=metrics.confusion_matrix(p_testY,y_pred)
    labels = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # create heatmap for confusion matrix
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("bottom")
    plt.title('Confusion matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')

    sensitivity=(cnf_matrix[1][1]/float(cnf_matrix[1][1]+cnf_matrix[1][0]))
    specificity=(cnf_matrix[0][0]/float(cnf_matrix[0][0]+cnf_matrix[0][1]))

    print("Accuracy:", metrics.accuracy_score(p_testY, y_pred))
    print("Sensitivity:",sensitivity)
    print("Specificity:",specificity)
    print("Recall:", metrics.recall_score(p_testY, y_pred))
    print("Precision:", metrics.precision_score(p_testY, y_pred))    
    plt.show()

#8- K-NN kullanarak veri setini sınıflandırın
#(Sonucun karmaşıklık matrisini çizdirin, bu matrise göre Accuracy, Sensitivity, Specificity, Recall, Precision değerlendirme kriterlerini hesaplayın.)
def Knn(p_trainX,p_testX,p_trainY,p_testY):
    model=KNeighborsClassifier(n_neighbors=5)
    model.fit(p_trainX,p_trainY)
    y_pred=model.predict(X_test)

    cnf_matrix=metrics.confusion_matrix(p_testY,y_pred)
    labels = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # create heatmap for confusion matrix
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("bottom")
    plt.title('Confusion matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')

    sensitivity=(cnf_matrix[1][1]/float(cnf_matrix[1][1]+cnf_matrix[1][0]))
    specificity=(cnf_matrix[0][0]/float(cnf_matrix[0][0]+cnf_matrix[0][1]))

    print("Accuracy:", metrics.accuracy_score(p_testY, y_pred))
    print("Sensitivity:",sensitivity)
    print("Specificity:",specificity)
    print("Recall:", metrics.recall_score(p_testY, y_pred))
    print("Precision:", metrics.precision_score(p_testY, y_pred))    
    plt.show()

#9- Naive Bayes kullanarak veri setini sınıflandırın
#(Sonucun karmaşıklık matrisini çizdirin, bu matrise göre Accuracy, Sensitivity, Specificity, Recall, Precision değerlendirme kriterlerini hesaplayın.)
def NaiveBayes(p_trainX,p_testX,p_trainY,p_testY):
    model=GaussianNB()
    model.fit(p_trainX,p_trainY)
    y_pred=model.predict(X_test)


    cnf_matrix=metrics.confusion_matrix(p_testY,y_pred)
    labels = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # create heatmap for confusion matrix
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("bottom")
    plt.title('Confusion matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')

    sensitivity=(cnf_matrix[1][1]/float(cnf_matrix[1][1]+cnf_matrix[1][0]))
    specificity=(cnf_matrix[0][0]/float(cnf_matrix[0][0]+cnf_matrix[0][1]))

    print("Accuracy:", metrics.accuracy_score(p_testY, y_pred))
    print("Sensitivity:",sensitivity)
    print("Specificity:",specificity)
    print("Recall:", metrics.recall_score(p_testY, y_pred))
    print("Precision:", metrics.precision_score(p_testY, y_pred))    
    plt.show()


if __name__=="__main__":
    df=pd.read_csv('data.csv')
    #Hasta_Dagilim(df)
    #Hasta_Cinsiyet_Dagilim(df)
    X_train, X_test, y_train, y_test=Data_Preprocess(df)
    #Logistic(X_train, X_test, y_train, y_test)
    #Knn(X_train, X_test, y_train, y_test)
    NaiveBayes(X_train, X_test, y_train, y_test)