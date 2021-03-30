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

# Plots the Sickness Distribution
def Sick_Distribution(p_df):
    status=[0,0]
    status[0]=len(p_df[p_df["target"]==1])
    status[1]=len(p_df)-status[0]
    text="Sick Count: "+str(status[0])+" | Healty Count: "+str(status[1])
    labels=['Sick','Healty']
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, status, align='center', alpha=0.5,color=['red','blue'])
    plt.xticks(y_pos, labels)
    plt.title('Sick Distribution')
    plt.figtext(0.5, 0.01, text, ha="center", fontsize=10,bbox={"facecolor":"white","alpha":0.5, "pad":5})
    plt.show()

#Plots the sickeness distrubution by gender
def Sick_Gender_Distribution(p_df):
    male_count=len(p_df[p_df["sex"]==1])
    sick_male_count=len(p_df[(p_df["sex"]==1)&(p_df["target"]==1)])
    female_count=len(p_df[p_df["sex"]==0])
    sick_female_count=len(p_df[(p_df["sex"]==0)&(p_df["target"]==1)])
    #sick_male,healty_male,sick_female,healty_female
    status=[sick_male_count,male_count-sick_male_count,sick_female_count,female_count-sick_female_count]    
    labels=['Male Sick','Male Healty','Female Sick','Female Healty']
    text=""
    for i in range(len(labels)):
        text+=str(labels[i])+": "+str(status[i])+" | "
    text=text[:-2]
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, status, align='center', alpha=0.5,color=['red','blue','red','blue'])
    plt.xticks(y_pos, labels)
    plt.title('Sick Gender Distribution')
    plt.figtext(0.5, 0.01, text, ha="center", fontsize=10,bbox={"facecolor":"white","alpha":0.5, "pad":5})
    plt.show()

#Plots the age distribution if paramater true else plots  sickness age distribution
def Age_Distribution(p_df,p_sick_status):
    plt_title="Age Distribution"
    if p_sick_status==True:
        p_df=p_df[p_df["target"]==1]
        plt_title="Sick Age Distribution"
    age_distribution=[]
    age_distribution.append(len(p_df[p_df["age"]<30]))
    for i in range(30,70,10):
        age_distribution.append(len(p_df[(p_df["age"]>=i)&(p_df["age"]<i+10)]))
    age_distribution.append(len(p_df[p_df["age"]>=70]))
    labels=["0-29","30-39","40-49","50-59","60-69","70+"]
    text=""
    for i in range(len(labels)):
        text+=str(labels[i])+": "+str(age_distribution[i])+" | "
    text=text[:-2]
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, age_distribution, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.title(plt_title)
    plt.figtext(0.5, 0.01, text, ha="center", fontsize=10,bbox={"facecolor":"white","alpha":0.5, "pad":5})
    plt.show()

#Data Preprocess %20 test
def Data_Preprocess(p_df):
    min_max_scaler=preprocessing.MinMaxScaler()
    data_scaled=min_max_scaler.fit_transform(p_df.values)
    df=pd.DataFrame(data_scaled,columns=p_df.columns)
    labels=df.target.copy()
    features=df.drop(["target"],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=256)
    return X_train, X_test, y_train, y_test

# Creates heatmap and claculates metrics by test data prediction
def Evaluate_Metrics(p_testY,y_pred,p_option):
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

    TP=cnf_matrix[0][0]
    FP=cnf_matrix[0][1]
    FN=cnf_matrix[1][0]
    TN=cnf_matrix[1][1]
    accuracy=(TP+TN)/(TP+FP+TN+FN)
    sensitivity=TP/(TP+FN)
    specificity=TN/(TN+FP)
    recall=sensitivity
    precision=TP/(TP+FP)

      
    plt.show()

    return (accuracy,sensitivity,specificity,recall,precision)


#Classifier for Logistic Reg, K-NN, Naive Bayes, Decision Tree
def Classifier(p_trainX,p_testX,p_trainY,p_testY,p_option):
    if p_option=="Logistic":
        model=LogisticRegression()
    elif p_option=="KNN":
        model=KNeighborsClassifier(n_neighbors=5)
    elif p_option=="NaiveBayes":
        model=GaussianNB()
    elif p_option=="DecisionTree":
        model=DecisionTreeClassifier()
    elif p_option=="NeuralNet":
        return Neural(p_trainX,p_testX,p_trainY,p_testY)      

    model.fit(p_trainX,p_trainY)
    y_pred=model.predict(X_test)
    return Evaluate_Metrics(p_testY,y_pred,p_option)
    

#Classifier for Neuralnet
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
    model.fit(p_trainX, p_trainY, validation_data=(p_testX, p_testY), epochs=50, verbose=2)
    scores = model.evaluate(p_testX, p_testY)
    predictions = model.predict(p_testX)
    
    y_pred=[]
    for i in predictions:
        y_pred.append(np.argmax(i))    

    return Evaluate_Metrics(dummy_y,y_pred,'NeuralNet')

#Plots the metric scores
def Plot(p_values,p_metric):
    if p_metric=="f1":
        f1=[]
        for i in p_values:
            f1.append((2*i[4]*i[3])/(i[4]+i[3]))
        index=["f1"]
        p_values=f1
    else:
        index=["Accuracy","Sensitivity","Specificity","Recall","Precision"]
    columns=["Logistic","KNN","NaiveBayes","DecisionTree","NeuralNet"]
    df=pd.DataFrame(np.column_stack(p_values),index=index,columns=columns)
    print(df)
    
    ypos=np.arange(len(columns))
    plt.xticks(ypos,columns)
    label=p_metric+" Score"
    plt.ylabel(label)
    plt.title(p_metric)
    plt.bar(ypos,df.loc[p_metric].values,align='center', alpha=0.5,color=['red','blue','orange','purple','green'])
    plt.show()

if __name__=="__main__":
    df=pd.read_csv('data.csv')

    Sick_Distribution(df)
    Sick_Gender_Distribution(df)
    Age_Distribution(df,False)
    Age_Distribution(df,True)

    X_train, X_test, y_train, y_test=Data_Preprocess(df)
    data=[X_train, X_test, y_train, y_test]

    metric_values=[]
    metric_values.append(Classifier(*data,"Logistic"))
    metric_values.append(Classifier(*data,"KNN"))
    metric_values.append(Classifier(*data,"NaiveBayes"))
    metric_values.append(Classifier(*data,"DecisionTree"))
    metric_values.append(Classifier(*data,"NeuralNet"))
    
    Plot(metric_values,"Accuracy")
    '''
    Plot(metric_values,"Sensitivity")
    Plot(metric_values,"Specificity")
    Plot(metric_values,"Recall")
    Plot(metric_values,"Precision")
    Plot(metric_values,"f1")
'''