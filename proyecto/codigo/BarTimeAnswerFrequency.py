import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/media/jhomara/Datos/MG-DCC/MD/dm_project/jp_question_answer2.csv", quoting=2,
                 usecols=['a_comment_count','a_creation_date', 'a_edit','a_score',
                          'answer_acepted','q_answer_count','q_comment_count',
                          'q_creation_date','q_favorite_count','q_score','q_view_count','answer_selected',
                          'user_reputation', 'user_up_votes', 'user_down_votes', 'user_views','first_answer_date'])
data['a_creation_date']=pd.to_datetime(data['a_creation_date'])
data['q_creation_date']=pd.to_datetime(data['q_creation_date'])
data['first_answer_date']=pd.to_datetime(data['first_answer_date'])
data['time_first_answer']=(data['first_answer_date']-data['q_creation_date']).fillna(0).astype('timedelta64[h]')
data['time_accepted_answer']=(data['a_creation_date']-data['q_creation_date']).fillna(0).astype('timedelta64[h]')

plt.hist(data['time_accepted_answer'], bins=10000, color="blue")
plt.ylim(ymax=5000)
plt.xlim(xmin=0,xmax=1000)
plt.legend("Respuesta Aceptada")
plt.title("Tiempo respuesta aceptada")
plt.ylabel("Frequencia")
plt.xlabel("Tiempo en horas")
plt.show()


plt.hist(data['time_first_answer'], bins=10000, color="red")
plt.ylim(ymax=5000)
plt.xlim(xmin=0,xmax=1000)
plt.ylabel("Frequencia")
plt.xlabel("Tiempo en horas")
plt.legend("Primera respuesta")
plt.title("Tiempo primera respuesta")
plt.show()
