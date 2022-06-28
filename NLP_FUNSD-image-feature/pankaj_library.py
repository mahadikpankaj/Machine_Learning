import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt

from datasets import load_dataset

# this function is used to download the train/test data from datasets
def get_data(data_type):
    global datasets_nielsr
    if 'datasets_nielsr' not in globals():
        datasets_nielsr = load_dataset('nielsr/funsd-image-feature')
        
    if not( data_type == 'train' or data_type=='test'):
        raise Error('train or test are the only valid values for data_type')
        
    all_words = [word for doc_words in datasets_nielsr[data_type]['words'] for word in doc_words]
    all_bboxes=[bbox for doc_bboxes in datasets_nielsr[data_type]['bboxes'] for bbox in doc_bboxes]
    df_words_all = pd.DataFrame(all_words, columns=['word'])
    df_bboxes_all = pd.DataFrame(all_bboxes, columns=['x1', 'y1', 'x2', 'y2'])
    
    all_ner_tags = [ner for doc_ner_tags in datasets_nielsr[data_type]['ner_tags'] for ner in doc_ner_tags]
    all_ner_tags = list(map(set_ner_tags, all_ner_tags))
    df_ner_tags_all = pd.DataFrame(all_ner_tags, columns=['ner'])
    
    df_all = pd.concat([df_words_all, df_bboxes_all, df_ner_tags_all], axis=1 )
    
    return df_all



# this function is used to map the B-Header and I-Header to Header, B-Question and I-Question to Question,
#  B-Answer and I-Answer to Answer
def set_ner_tags(ner_tag):
    ner=0
    if ner_tag in [1, 2]:
        ner = 1
    if ner_tag in [3, 4]:
        ner = 2
    if ner_tag in [5, 6]:
        ner = 3
    return ner






# this function checks and marks if the word contains colon or slash in it
def set_has_colon(word):
    ret = 0
    if ':' in word:
        ret = 1
    elif '/' in word:
        temp = word.replace('/', '').replace(' ', '')
        if temp.isalpha():
            ret = 1
        if temp.isnumeric():
            ret = 0
    return ret





# function to check if the word is in capital, small or mixed case letters
def set_lower_or_upper(word):
    case_type = 1
    if word == word.lower():
        case_type = 0
    elif word == word.upper():
        case_type = 2
    return case_type







# function to generate the ground truth of labels identifying type of word
# 0 - Other
# 1 - Header
# 2 - Question
# 3 - Answer
def decide_set_ner_flags(word_rec):
    ner = 3
    if word_rec['has_colon']==1:
        ner = 2
    elif ',' in word_rec['word'] or '.' in word_rec['word']:
        ner = 3
    elif word_rec['char_case']==2:
        ner = 1
    elif word_rec['y2'] < 250:
        ner = 1
    return ner





# function to pre-process the data
def pre_prcess(df_curr):
    df_curr['word'] = df_curr['word'].apply(lambda x:x.strip())
    df_curr['has_colon'] = df_curr['word'].apply(set_has_colon)
    df_curr['word'] = df_curr['word'].apply(lambda x:x.replace(':', ''))
    df_curr['char_case'] = df_curr['word'].apply(set_lower_or_upper)
    df_curr['word'] = df_curr['word'].apply(lambda x:x.lower())
    df_curr['ner_calc'] = df_curr[['word', 'has_colon', 'char_case', 'y2']].apply(decide_set_ner_flags, axis=1)
    return df_curr
    
    
    
# function to process the data like one hot encoding, min-max scaling etc.    
def process(X_curr, is_train=False):
    global ohe_word
    global ohe_has_colon
    global ohe_char_case
    global X_train_word_list
    global min_max_scaler_cordinates
    
    if is_train:
        ohe_word = OneHotEncoder(sparse=False)
        ohe_has_colon = OneHotEncoder(sparse=False)
        ohe_char_case = OneHotEncoder(sparse=False)
        min_max_scaler_cordinates = MinMaxScaler()
        X_train_word_list = list(X_curr['word'])
        ohe_word = ohe_word.fit(X_curr[['word']])
        df_has_colon = ohe_has_colon.fit(X_curr[['has_colon']])
        df_char_case = ohe_char_case.fit(X_curr[['char_case']])
        
    if not(is_train):
        X_curr.loc[:, 'word'] = X_curr['word'].apply(lambda x: x if x in X_train_word_list else 'other')
            
    
    ohe_word_data = ohe_word.transform(X_curr[['word']])
    ohe_has_colon_data = ohe_has_colon.transform(X_curr[['has_colon']])
    ohe_char_case_data = ohe_char_case.transform(X_curr[['char_case']])
    df_ohe_word = pd.DataFrame(ohe_word_data, columns=ohe_word.get_feature_names()).astype(int)
    df_ohe_has_colon_word = pd.DataFrame(ohe_has_colon_data, columns=ohe_has_colon.get_feature_names()).astype(int)
    df_ohe_char_case = pd.DataFrame(ohe_char_case_data, columns=ohe_char_case.get_feature_names()).astype(int)
    
    if is_train:
        min_max_scaler_cordinates = min_max_scaler_cordinates.fit(X_curr[['x1', 'y1', 'x2', 'y2']])

    df_min_max_scaler_cordinates = pd.DataFrame(min_max_scaler_cordinates.transform(X_curr[['x1', 'y1', 'x2', 'y2']]), columns=['x1', 'y1', 'x2', 'y2'])
    X_curr_updated = pd.concat([df_ohe_word, df_ohe_has_colon_word, df_ohe_char_case, df_min_max_scaler_cordinates], axis=1)
    return X_curr_updated




# function to train the models
def train_models(model, X_train, y_train):
    model_name = str(model)
    model_name = model_name[0:model_name.index("(")]
    print('Fitting model: ' + model_name)
    model.fit(X_train, y_train)

    
    
    
# function to plot the confusion matrix of performance matrix    
def plot_confusion_matrix_for_models(model, X_test, y_test):
    model_name = str(model)
    model_name = model_name[0:model_name.index("(")]
    print('Plotting for model: ' + model_name)

    y_pred = pd.Series(model.predict(X_test))
    cm= metrics.confusion_matrix(y_test, y_pred)
    print('confusion matrix:')
    display(cm)
    mcm = metrics.multilabel_confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    print('multilabel confusion matrix:')
    display(mcm)
    metrics.plot_confusion_matrix(model, X_test, y_test, display_labels=['Other', 'Header', 'Question', 'Answer'], colorbar=False)
    plt.show()
    
    

# evaluate the models and print the performance metrics
def evaluate_models(models_list, X_test, y_test): 
    global df_performance_matrix
    df_performance_matrix = pd.DataFrame(columns=['accuracy', 'f1_score (header)', 'f1_score (question)', 'f1_score (answer)', 'precision (header)', 'precision (question)', 'precision (answer)'])
    for model in models_list:
        evaluate_model(model, X_test, y_test)    
    
    display(df_performance_matrix)

    
    
    
# actual method which evaluates the model and calculate various performance metrics    
def evaluate_model(model, X_test, y_test):
    global df_performance_matrix
    
    
    model_name = str(model)
    model_name = model_name[0:model_name.index("(")]
    print('Evaluating model: ' + model_name)

    y_pred = pd.Series(model.predict(X_test))
    
    accu = metrics.accuracy_score(y_test, y_pred)
    
    f1_score_1 = metrics.f1_score(y_test, y_pred, average='micro', labels=[1])
    f1_score_2 = metrics.f1_score(y_test, y_pred, average='micro', labels=[2])
    f1_score_3 = metrics.f1_score(y_test, y_pred, average='micro', labels=[3])
    
    precision_1 = metrics.precision_score(y_test, y_pred, average='micro', labels=[1])
    precision_2 = metrics.precision_score(y_test, y_pred, average='micro', labels=[2])
    precision_3 = metrics.precision_score(y_test, y_pred, average='micro', labels=[3])
    
    model_name = str(model)
    model_name = model_name[0:model_name.index("(")]
    df_performance_matrix.loc[model_name] = [accu, f1_score_1, f1_score_2, f1_score_3, precision_1, precision_2, precision_3] 
    
  
    
    
    
# function to predict the word provided as input    
def predict_word_type(models_list, to_predict):
    df_to_predict = pd.DataFrame(data=to_predict, index=[0])
    #display(df_to_predict)
    df_predict = pre_prcess(df_to_predict)
    #display(df_predict.head())
    df_predict.loc[:, :] = df_predict[X_columns]
    df_predict = process(df_predict, is_train=False)
    #display(df_predict.head())

    df_results = pd.DataFrame(columns=['Word', 'model name', 'Predicted Type'])
    i = 0
    for model in models_list:
        i+=1
        model_name = str(model)
        model_name = model_name[0:model_name.index("(")]
        df_results.loc[i] = [to_predict['word'], model_name, word_type_from_numeric[model.predict(df_predict)[0]]]

    display(df_results)
    #return df_results    
    
    
    
   

