import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv("/media/jhomara/Datos/MG-DCC/MD/dm_project/answer_count1.csv", quoting=2,
                 usecols=['answer_accepted','answer_count', 'questions'])

datax1=data[data['answer_accepted']==1]
datax2=data[data['answer_accepted']==0]
X1=datax1['answer_count'].values
x2=datax2['answer_count'].values

line_up=plt.bar(X1+0.3, datax1['questions'].values, color = 'r', width = 0.50, label='Con respuesta')
line_down=plt.bar(x2+0.7, datax2['questions'].values, color = 'b', width = 0.50, label='Sin respuesta')
plt.legend(handles=[line_up, line_down])
plt.xlabel('Cantidad de respuestas por pregunta')
plt.ylabel('Cantidad de preguntas')
plt.show()


#plt.show()