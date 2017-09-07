from pandas.plotting import parallel_coordinates
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/media/jhomara/Datos/MG-DCC/MD/dm_project/data.csv", quoting=2,
                 usecols=['q_answer_count','q_comment_count','q_creation_date','q_favorite_count',
                          'q_view_count','a_comment_count','a_edit','a_score',
                          'first_answer_date','last_answer_date','accepans_date','accepans_comment_count',
                          'accepans_a_edit','accepans_a_score','accepans'])


data['q_creation_date']=pd.to_datetime(data['q_creation_date'])
data['accepans_date']=pd.to_datetime(data['accepans_date'])
data['first_answer_date']=pd.to_datetime(data['first_answer_date'])
data['last_answer_date']=pd.to_datetime(data['last_answer_date'])
data['time_first_answer']=(data['first_answer_date']-data['q_creation_date']).fillna(0).astype('timedelta64[m]')
data['time_last_answer']=(data['last_answer_date']-data['q_creation_date']).fillna(0).astype('timedelta64[m]')
data['time_accepted_answer']=(data['accepans_date']-data['q_creation_date']).fillna(0).astype('timedelta64[m]')

xdata=data.loc[:, lambda df: ['q_answer_count','q_comment_count','q_favorite_count',
                          'q_view_count','a_comment_count','a_edit','a_score',
                          'time_first_answer',
                           'time_last_answer','time_accepted_answer',
                           'accepans_comment_count',
                           'accepans_a_edit', 'accepans_a_score']]
#print(xdata)

normalizer=preprocessing.Normalizer().fit(xdata)
norm_data=normalizer.transform(xdata)
xdf=pd.DataFrame(data=norm_data[0:,0:],    # values
                  # 1st column as index
              columns=['q_answer','q_comment','q_favorite',
                          'q_view','a_comment','a_edit','a_score',
                          'time_first_a',
                           'time_last_a','time_atd',
                           'atd_comments',
                           'atd_edit', 'atd_score'
                       ])
xdf.loc[:,'accepans']=data.loc[:,'accepans']
data_sample=xdf.sample(frac=0.1, replace=True)
parallel_coordinates(data_sample, 'accepans')
plt.show()

