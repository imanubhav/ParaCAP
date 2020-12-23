#training data
df = pd.read_table('../input/traindata/train.tsv',sep='\t')
df.columns=['qid','question','specification','target']
df=df[df.specification!='no answer']
df=df.loc[df['target'].isin([0,1])]
df["question_text"] = df["question"].map(str) + df["specification"]
df.drop(['qid','question', 'specification'], axis=1)
df = df[['question_text','target']]
df1=df.loc[df['target'] == 0]
df1=df1[:57138]
df2=df.loc[df['target']==1]
df2=df2
frames = [df1, df2]
train= pd.concat([df1,df2])
from sklearn.utils import shuffle
df = shuffle(df)
X = df.iloc[:,0:-1 ].values
y = df.iloc[:, -1].values

# testing data
df_test1 =pd.read_table('../input/testdata1/test_ac.tsv',sep='\t')
#df_test1 =pd.read_table('../input/testdata2/test_bkp.tsv',sep='\t')
#df_test1 =pd.read_table('../input/testdata3/test_com.tsv',sep='\t')
#df_test1 =pd.read_table('../input/testdata4/test_sho.tsv',sep='\t')
#df_test1 =pd.read_table('../input/testdata5/test_wat.tsv',sep='\t')

df_test1.columns=['id','question','specification','target']
df_test1=df_test1.sort_values('question', ascending=True)
df_test2=df_test1.groupby('question')['target'].apply(list)
a=list(df_test2)
b=list()
for i in range(len(a)):
    b.append(len(a[i]))
    
df_test3=pd.DataFrame()
df_test3["question_text"] = df_test1["question"].map(str) + df_test1["specification"]
df_test1.drop(['id','question', 'specification'], axis=1)
df_test3['target']=df_test1['target']
df_test3 = df_test3[['question_text','target']]
Xtest = df_test3.iloc[:,0:-1 ].values
ytest = df_test3.iloc[:, -1].values
X_train=X
X_test=Xtest
y_train=y
y_test=ytest
d1 = pd.DataFrame(X_train,columns=['question_text'])
d2=pd.DataFrame(y_train,columns=['target'])
df_train = pd.concat([d1, d2], axis=1, join_axes=[d1.index])
d3 = pd.DataFrame(X_test,columns=['question_text'])
d4=pd.DataFrame(y_test,columns=['target'])
df_test = pd.concat([d3, d4], axis=1, join_axes=[d3.index])

