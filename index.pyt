import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB

data = pd.read_csv('dataFix.csv')

column = ['Age(Years)','Sleep time (Hours)','Time spent on social media (Hours)','Study time (Hours)']
X = np.array(data[column])
Y = np.array(data['Your level of satisfaction in Online Education'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# proses klasifikasi KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)

# hasil klasifikasi KNN
Y_pred = knn.predict(X_test)
print(Y_pred)

# dataframe kelas aktual dan kelas sistem
df_datahasilknn = pd.DataFrame(X_test,columns=['Age(Years)','Sleep time (Hours)','Time spent on social media (Hours)','Study time (Hours)'])
df_datahasilknn['Kelas Aktual']=Y_test
df_datahasilknn['Kelas Sistem']=np.array(Y_pred)
print(df_datahasilknn)

# hasil evaluasi KNN
print(confusion_matrix(Y_test, Y_pred))
print('----- Hasil Evaluasi Data -----')
print(' ')
print('Nilai Akurasi: {:.2f}'.format(accuracy_score(Y_test, Y_pred)))
print('Nilai Presisi: {:.2f}'.format(precision_score(Y_test,Y_pred, average='weighted')))
print('Nilai Recall: {:.2f}'.format(recall_score(Y_test, Y_pred,average='weighted')))
print('Nilai F1: {:.2f}'.format(f1_score(Y_test, Y_pred,average='weighted')))
hasilpengujian=classification_report(Y_test,Y_pred)
print(hasilpengujian)

# proses klasifikasi KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)

# hasil klasifikasi KNN
Y_pred = knn.predict(X_test)
print(Y_pred)

# dataframe kelas aktual dan kelas sistem
df_datahasilknn = pd.DataFrame(X_test,columns=['Age(Years)','Sleep time (Hours)','Time spent on social media (Hours)','Study time (Hours)'])
df_datahasilknn['Kelas Aktual']=Y_test
df_datahasilknn['Kels Sistem']=np.array(Y_pred)
print(df_datahasilknn)

# hasil evaluasi KNN
print(confusion_matrix(Y_test, Y_pred))
print('----- Hasil Evaluasi Data -----')
print(' ')
print('Nilai Akurasi: {:.2f}'.format(accuracy_score(Y_test, Y_pred)))
print('Nilai Presisi: {:.2f}'.format(precision_score(Y_test, Y_pred, average='weighted')))
print('Nilai Recall: {:.2f}'.format(recall_score(Y_test, Y_pred, average='weighted')))
print('Nilai F1: {:.2f}'.format(f1_score(Y_test, Y_pred, average='weighted')))
hasilpengujian=classification_report(Y_test,Y_pred)
print(hasilpengujian)

# proses klasifikasi naive bayes
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

#Hasil klasifikasi naive bayes
nilaiprob = classifier.predict_proba(X_test)
print(nilaiprob)
Y_pred2 = classifier.predict(X_test)
print(Y_pred2)

# dataframe kelas aktual dan kelas sistem
df_datahasilnaive = pd.DataFrame(X_test,columns=['Age(Years)','Sleep time (Hours)','Time spent on social media (Hours)','Study time (Hours)'])
df_datahasilnaive['Kelas Aktual']=Y_test
df_datahasilnaive['Kelas Sistem']=np.array(Y_pred2)
print(df_datahasilnaive)

# hasil evaluasi naive bayes
print(confusion_matrix(Y_test, Y_pred2))
print('----- Hasil Evaluasi Data -----')
print(' ')
print('Nilai Akurasi: {:.2f}'.format(accuracy_score(Y_test, Y_pred2)))
print('Nilai Presisi: {:.2f}'.format(precision_score(Y_test, Y_pred2, average='weighted')))
print('Nilai Recall: {:.2f}'.format(recall_score(Y_test, Y_pred2, average='weighted')))
print('Nilai F1: {:.2f}'.format(f1_score(Y_test, Y_pred2, average='weighted')))
hasilpengujian2=classification_report(Y_test,Y_pred2)
print(hasilpengujian2)