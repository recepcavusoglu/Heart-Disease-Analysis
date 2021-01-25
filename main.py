import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

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
        p_df=p_df[p_df["target"]==1]
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
    
#7-Logistic Regresyon kullanarak veri setini sınıflandırın
#8- K-NN kullanarak veri setini sınıflandırın
#9- Naive Bayes kullanarak veri setini sınıflandırın
#10- Karar ağaçları kullanarak veri setini sınıflandırın
#(Sonucun karmaşıklık matrisini çizdirin, bu matrise göre Accuracy, Sensitivity, Specificity, Recall, Precision değerlendirme kriterlerini hesaplayın.)
def Classifier(p_trainX,p_testX,p_trainY,p_testY,p_option):
    if p_option=="Logistic":
        model=LogisticRegression()
    elif p_option=="KNN":
        model=KNeighborsClassifier(n_neighbors=5)
    elif p_option=="NaiveBayes":
        model=GaussianNB()
    elif p_option=="DecisionTree":
        model=DecisionTreeClassifier()

    model.fit(p_trainX,p_trainY)
    y_pred=model.predict(X_test)
    cnf_matrix=metrics.confusion_matrix(p_testY,y_pred)
    labels = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("bottom")
    title=p_option+' Confusion matrix'
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')

    accuracy=metrics.accuracy_score(p_testY, y_pred)
    sensitivity=(cnf_matrix[1][1]/float(cnf_matrix[1][1]+cnf_matrix[1][0]))
    specificity=(cnf_matrix[0][0]/float(cnf_matrix[0][0]+cnf_matrix[0][1]))
    recall=metrics.recall_score(p_testY, y_pred)
    precision=metrics.precision_score(p_testY, y_pred)

    print(p_option," Accuracy:", accuracy)
    print(p_option," Sensitivity:",sensitivity)
    print(p_option," Specificity:",specificity)
    print(p_option," Recall:", recall)
    print(p_option," Precision:", precision)    
    plt.show()
    return (accuracy,sensitivity,specificity,recall,precision)

#11- Yapay Sinir Ağları kullanarak veri setini sınıflandırın
#(Sonucun karmaşıklık matrisini çizdirin, bu matrise göre Accuracy, Sensitivity, Specificity, Recall, Precision değerlendirme kriterlerini hesaplayın.)
def Neural(p_trainX,p_testX,p_trainY,p_testY):
    dummy_y=p_testY
    p_trainY=to_categorical(p_trainY)
    p_testY=to_categorical(p_testY)

    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(p_trainX, p_trainY, validation_data=(p_testX, p_testY), epochs=200, verbose=2)
    scores = model.evaluate(p_testX, p_testY)
    predictions = model.predict(p_testX)
    
    y_pred=[]
    for i in predictions:
        y_pred.append(np.argmax(i))    

    cnf_matrix=metrics.confusion_matrix(dummy_y,y_pred)
    labels = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("bottom")
    plt.title('NN Confusion matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')

    accuracy=metrics.accuracy_score(dummy_y, y_pred)
    sensitivity=(cnf_matrix[1][1]/float(cnf_matrix[1][1]+cnf_matrix[1][0]))
    specificity=(cnf_matrix[0][0]/float(cnf_matrix[0][0]+cnf_matrix[0][1]))
    recall=metrics.recall_score(dummy_y, y_pred)
    precision=metrics.precision_score(dummy_y, y_pred)

    print("NN Accuracy:", accuracy)
    print("NN Sensitivity:",sensitivity)
    print("NN Specificity:",specificity)
    print("NN Recall:", recall)
    print("NN Precision:", precision)   
    plt.show()
    return (accuracy,sensitivity,specificity,recall,precision)

def Plot(p_values):
    '''
    labels=["Accuracy","Sensitivity","Specificity","Recall","Precision"]
    y_pos = np.arange(len(labels))
    plt.bar(y_pos,p_values, align='center', alpha=0.5,color=["orange","blue","red","purple","yellow"])
    plt.xticks(y_pos, labels)
    plt.title("asdas")
    plt.show()
    #pd.crosstab(p_values[0],p_values[1],p_values[2],p_values[3],p_values[4]).plot(kind="bar",figsize=(15,6))
    '''
    df=pd.DataFrame(p_values,index=["accuracy","sensitivity","specificity","recall","precision"],columns=["Logistic","KNN","NaiveBayes","DecisionTree","NeuralNet"])
    print(df)

    

if __name__=="__main__":
    df=pd.read_csv('data.csv')
    #Hasta_Dagilim(df)
    #Hasta_Cinsiyet_Dagilim(df)
    #Yas_Dagilimi(df,True)
    X_train, X_test, y_train, y_test=Data_Preprocess(df)
    data=[X_train, X_test, y_train, y_test]
    metric_values=[]
    metric_values.append(Classifier(*data,"Logistic"))
    metric_values.append(Classifier(*data,"KNN"))
    metric_values.append(Classifier(*data,"NaiveBayes"))
    metric_values.append(Classifier(*data,"DecisionTree"))
    metric_values.append(Neural(*data))
    Plot(metric_values)