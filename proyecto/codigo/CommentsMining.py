import pandas as pd
import matplotlib.pyplot as plt
# To read a CSV file
# df = pd.read_csv('sentences.csv')
df = pd.read_csv("/media/jhomara/Datos/MG-DCC/MD/dm_project/question_comments.csv", quoting=2,)

from textblob import TextBlob

# The x in the lambda function is a row (because I set axis=1)
# Apply iterates the function accross the dataframe's rows
def delete_replace(s,fill_char = " "):
    return "".join([x  if x.isalpha() else fill_char for x in s])

df['text_clean'] = df['comments_text'].apply(lambda x: delete_replace(x))#
df['subjectivity'] = df.apply(lambda x: TextBlob(x['text_clean']).sentiment.polarity, axis=1)
df['polarity'] = df.apply(lambda x: TextBlob(x['text_clean']).sentiment.subjectivity, axis=1)
"""df['positive'] = df['polarity'].apply(lambda x: 1 if x>0 else 0)
df['negative'] = df['polarity'].apply(lambda x: 1 if x<0 else 0)
df['neutral'] = df['polarity'].apply(lambda x: 1 if x==0 else 0)
#df['subjectivity'] = df.apply(lambda x: TextBlob(x['comments_text']).sentiment.subjectivity, axis=1)

dfFilter=df.loc[:, lambda df: ['q_id','accepted_answer','positive','negative','neutral']]

#index = pd.MultiIndex.from_arrays(dfFilter, names=['q_id','accepted_answer'])

#dfIndex=pd.DataFrame(dfFilter,index)

#dfsum=dfIndex.groupby(level=['q_id','accepted_answer'])

dfAnsweredPP=pd.DataFrame(dfFilter[dfFilter['accepted_answer'] == 1].groupby('q_id')['positive'].sum().reset_index(name='positiveA'))
dfAnsweredPN=pd.DataFrame(dfFilter[dfFilter['accepted_answer'] == 1].groupby('q_id')['negative'].sum().reset_index(name='negativeA'))
dfNAnsweredPP=pd.DataFrame(dfFilter[dfFilter['accepted_answer'] == 0].groupby('q_id')['positive'].sum().reset_index(name='positiveNa'))
dfNAnsweredPN=pd.DataFrame(dfFilter[dfFilter['accepted_answer'] == 0].groupby('q_id')['negative'].sum().reset_index(name='negativeNa'))
dfAnsweredPNeu=pd.DataFrame(dfFilter[dfFilter['accepted_answer'] == 1].groupby('q_id')['neutral'].sum().reset_index(name='neutralA'))
dfNAnsweredPNeu=pd.DataFrame(dfFilter[dfFilter['accepted_answer'] == 0].groupby('q_id')['neutral'].sum().reset_index(name='neutralNa'))


dfAnsweredPP.set_index('q_id')
dfAnsweredPN.set_index('q_id')
dfNAnsweredPP.set_index('q_id')
dfNAnsweredPN.set_index('q_id')
dfAnsweredPNeu.set_index('q_id')
dfAnsweredPNeu.set_index('q_id')
dfNAnsweredPNeu.set_index('q_id')
dfNAnsweredPNeu.set_index('q_id')
#frames=[dfAnsweredPP,dfAnsweredPN,dfNAnsweredPP,dfNAnsweredPN]

#res2=pd.concat(frames)
res1=pd.merge(dfAnsweredPP,dfAnsweredPN)
res2=pd.merge(dfNAnsweredPP,dfNAnsweredPN)
res3=pd.merge(dfAnsweredPNeu,dfNAnsweredPNeu)
merge1=pd.merge(res1,res2, how='outer')
merge2=pd.merge(merge1,res3, how='outer')

merge2.set_index('q_id')

merge2.groupby


#sumaSP=dfsum.sum()
print(merge2)
"""
data_sample=df.sample(frac=0.1, replace=True)

X=df.loc[:, lambda df: ['subjectivity','polarity']]
y=df['accepted_answer']

plt.scatter(X[y==0]['subjectivity'], X[y==0]['polarity'], label='Sin respuesta', c='red')
plt.scatter(X[y==1]['subjectivity'], X[y==1]['polarity'], label='Con respuesta', c='blue')

plt.legend()
plt.xlabel('subjectivity')
plt.ylabel('polarity')

# display
plt.show()