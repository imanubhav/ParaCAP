#predicting 
y_pred=model.predict(x_test)
y_pred1=list(y_pred)

def softmax(x):
    
    return (np.exp(x) / np.sum(np.exp(x), axis=0))

y_pred2=softmax(y_pred)
y_pred3=y_pred2.ravel()

m=pd.DataFrame(
    {'probability': y_pred3,
     
     'actual': y_test
    })
    
    
#create sorted nested list
from itertools import islice 
Inputt = iter(y_pred2) 
Output = [list(islice(Inputt, elem)) 
          for elem in b] 
for sublist in Output: 
    sublist.sort(reverse=True) 

d=list()
e=list()
f=list()
g=list()
h=list()
for i in range(len(Output)):
    d.append(Output[i][0])
for i in range(len(Output)):
    e.append(Output[i][1])
for i in range(len(Output)):
    f.append(Output[i][2])
for i in range(len(Output)):
    g.append(Output[i][3])
for i in range(len(Output)):
    h.append(Output[i][4])
    

m['@1'] = np.where((m.probability.isin(d)), 1, 10)
m['@2'] = np.where((m.probability.isin(e)), 1, 11)
m['@3'] = np.where((m.probability.isin(f)), 1, 12)
m['@4'] = np.where((m.probability.isin(g)), 1, 13)
m['@5'] = np.where((m.probability.isin(h)), 1, 14)

m=m.drop_duplicates()

#Calculate number of hits
l1=len(m.loc[m['@1'] == m['actual']])
l2=len(m.loc[m['@2'] == m['actual']])
l3=len(m.loc[m['@3'] == m['actual']])
l4=len(m.loc[m['@4'] == m['actual']])
l5=len(m.loc[m['@5'] == m['actual']])
print('For @1:    ', l1)
print('For @2:    ',l2)
print('For @3:    ',l3)
print('For @4:    ',l4)
print('For @5:    ',l5)
print(len(b))

#Calculate hit score
print('For @1:    ',l1/len(b))
print('For @2:    ',(l1+l2)/len(b))
print('For @3:    ',(l1+l2+l3)/len(b))
print('For @4:    ',(l1+l2+l3+l4)/len(b))
print('For @5:    ',(l1+l2+l3+l4+l5)/len(b))

#calculate confusion matrix
y_pred2.astype(int)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())




