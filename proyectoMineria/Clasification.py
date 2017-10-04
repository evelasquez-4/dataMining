from bs4 import BeautifulSoup
import pandas as pd

df=pd.read_csv("/media/jhomara/Datos/MG-DCC/MINERIA_DATOS/dm_project/entrega2/question_example2.csv", quotechar='"',
               usecols=['post_id','accepted_ans','post_title','post_body','post_creation_date',
                            'post_answer_count','post_comment_count',
                            'post_favorite_count', 'post_score','post_tags',
                            'post_view_count','users_creation_date',
                            'users_reputation',	'users_up_votes','users_down_votes',
                            'score_prev_acceptans' ,'score_prev_ans',
                            'score_prev_comment','score_prev_question',
                            'score_prev_favquestion'])
df_tags=pd.read_csv("/media/jhomara/Datos/MG-DCC/MINERIA_DATOS/dm_project/entrega2/top_tags.csv", quotechar='"')

df['post_creation_date']=pd.to_datetime(df['post_creation_date'])
df['users_creation_date']=pd.to_datetime(df['users_creation_date'])
df2 = pd.DataFrame(data=df, index=df.index, columns=['accepted_ans',
                            'post_comment_count',
                            'post_favorite_count', 'post_score',
                            'post_view_count',
                            'users_reputation',	'users_up_votes','users_down_votes',
                            'score_prev_acceptans' ,'score_prev_ans',
                            'score_prev_comment','score_prev_question',
                            'score_prev_favquestion']
                   )
df2['age_user'] = (df['post_creation_date'] - df['users_creation_date']).fillna(0).astype('timedelta64[D]')
df2['title_length'] = df['post_title'].apply(lambda x: len(x))
df2['num_block_code'] = 0
df2["num_i_sentences"]=0
df2["num_wh_words"]=0
df2["num_y_sentences"]=0
df2["tags_popularity"]=0
df2["num_tags"]=0

whwords=['what','how', 'which', 'when', 'why', 'where']
for index, row in df.iterrows():
    sbody=row["post_body"]
    soup = BeautifulSoup(sbody, "html5lib")
    #isentences = soup.find_all(name="p", text=re.compile('^I'))
    sentences =  soup.find_all(name="p")
    count_wh=0
    count_is=0
    count_ys = 0
    for sentence in sentences:

        try:
            count_is = count_is + len([x for x in sentence.contents[0].split() if (x == "I")])
            count_is = count_is + (len(sentence.contents[0].split("I'")) - 1)
        except:
            count_is = count_is + len([x for x in str(sentence.contents).split() if (x == "I")])
            count_is = count_is + (len(str(sentence.contents).split("I'")) - 1)

        try:
            count_ys = count_ys + len([x for x in sentence.contents[0].split() if (x == "you")])
            count_ys = count_ys + (len(sentence.contents[0].split("you'")) - 1)
        except:
            count_ys = count_ys + len([x for x in str(sentence.contents).split() if (x == "you")])
            count_ys = count_ys + (len(str(sentence.contents).split("you'")) - 1)

        for word in whwords:
            try:
                count_wh=count_wh+len([x for x in sentence.contents[0].split() if x == word])
            except:
                count_wh = count_wh + len([x for x in str(sentence.contents).split() if x == word])
    df2.loc[index, "num_i_sentences"] = count_is
    df2.loc[index, "num_wh_words"] = count_wh
    df2.loc[index, "num_y_sentences"] = count_ys

for index, row in df.iterrows():
    body=row["post_body"]
    tags=row["post_tags"]
    counttag=len(tags.split("|"))
    soup = BeautifulSoup(body, "html5lib")
    precode = soup.find_all("pre")
    df2.loc[index, "num_block_code"]=len(precode)
    content=""
    countError = 0
    for codeline in precode:
        contentPre = codeline.contents
        for contentCode in contentPre:
            try:
                content=content+contentCode.contents[0]
            except :
                content = content + contentCode
    wordCodeCount =len(content)
    df2.loc[index,"code_length"]=wordCodeCount
    df2.loc[index, "tag_count"] = counttag
#print(df2)

#print(df_tags['tag_name'].values)

for index, row in df.iterrows():
    tags_column=row["post_tags"]
    tags=tags_column.split("|")
    counttag=len(tags)
    pop_tag=0
    for tag in tags:
        if tag in df_tags['tag_name'].values:
            pop_tag+=1;
    df2.loc[index, "num_tags"] = counttag
    df2.loc[index, "tags_popularity"] = pop_tag
#print(df2)

df2.to_csv(path_or_buf='/media/jhomara/Datos/MG-DCC/MINERIA_DATOS/dm_project/entrega2/features.csv', sep=str(u','), quotechar='"')

"""
num_features = ['post_comment_count',
                'post_favorite_count', 'post_score',
                'post_view_count',
                'users_reputation',	'users_up_votes','users_down_votes',
                'score_prev_acceptans' ,'score_prev_ans',
                'score_prev_comment','score_prev_question',
                'score_prev_favquestion','code_length','tag_count','age_user','title_length']

scaled_features = {}
for each in num_features:
    mean, std = df2[each].mean(), df2[each].std()
    scaled_features[each] = [mean, std]
    df2.loc[:, each] = (df2[each] - mean)/std

features=df2.values[:,:18]
target=df2.values[:,0]
#print(target)
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target, test_size = 0.33, random_state = 10)
clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)

accuracy =accuracy_score(target_test, target_pred, normalize = True)
cnf_matrix = confusion_matrix(target_test, target_pred)
print(cnf_matrix)
"""
