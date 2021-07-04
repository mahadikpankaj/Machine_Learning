import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fillna_cont_var_with_random_ranged_values(df, col, method='mean'):
    v_data = df.agg({col:[method, 'std']})[col]
    v_middle = v_data[method]
    v_std = v_data['std']
    v_null_count = df[col].isna().sum()
    v_rand_values = np.random.randint(v_middle - v_std, v_middle + v_std, size = v_null_count)
    v_slice = df[col].copy()
    v_slice[np.isnan(v_slice)] = v_rand_values
    df[col] = v_slice
    df[col] = df[col].astype(int)
	
def get_categorical_continuous_variable_names (df):
    col_cate=list()
    col_cont=list()

    for x, y in df.dtypes.to_dict().items():
        if str(y) in ['object', 'category']:
            col_cate.append(x.strip())
        else:
            col_cont.append(x.strip())
    return {'col_cont':sorted(col_cont), 'col_cate':sorted(col_cate)}
	
def get_categories_to_consider(data, col, threshold=10):
    cat_list=[]
    count_per = data[col].count()
    for x,y in data.groupby([col]).size().sort_values(ascending=False).to_dict().items():
        if y/count_per*100 >= threshold:
            cat_list.append(x)
        else:
            break
    return sorted(cat_list[:20])
	
def labelize_column_data(df, col, threshold=10):
    cat_values = get_categories_to_consider(df, col, threshold)
    df[col] = df[col].apply(lambda x: x if x in cat_values else 'Others').astype(str)
	
	
def encode_df_column_per_frequency(df_train, df_test, col):
    name_count_dict_1 = df_train.groupby([col]).size().sort_values().to_dict()
    key_dict={}
    for x, y in name_count_dict_1.items():
        key_list = key_dict.get(y)
        if key_list is None:
            key_list = list()
            key_dict[y] = key_list
        key_list.append(x)
    name_count_dict = {y:x for x, y in enumerate(key_dict)}
    name_count_dict_2 = {}
    for x, y in list(enumerate(key_dict)):
        for k in key_dict[y]:
            name_count_dict_2[k] = x
    df_train[col] = df_train[col].apply(lambda x: name_count_dict_2[x])
    df_test[col] = df_test[col].apply(lambda x: name_count_dict_2.get(x))
	
def normalize_with_standard_scaler(df, col):
    std_scaler = StandardScaler()
    df[col] = std_scaler.fit_transform(df[col].values.reshape(-1, 1))
	
def predict_calculate_print_accuracy(model, X_train, X_test, y_train, y_test):
    model_name = str(model).split('(')[0]
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(model_name)
    print(f"Model Accuracy: {round(model.score(X_train, y_train) * 100.0, 2)}")
    print(f"Prediction Accuracy: {round(accuracy * 100.0, 2)}")
    print()
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)
    print()
    print(classification_report(y_test, y_pred))
    ax = sns.heatmap(cf_matrix,annot=True,fmt='3.0f',cmap="RdYlGn")
    ax.set_title(model_name)
    plt.show()
	
def predict_and_output_results(model, file_path):
    result = model.predict(test_data)
    result_df = pd.DataFrame(result, columns=['Survived'])
    final_result_df = pd.concat([df_test_orig[['PassengerId']], result_df], axis=1)
    final_result_df.to_csv(file_path, index=False)


def plot_heat_map(df, figsize=(10,10)):
    plt.figure(figsize=figsize)
    #fig=plt.gcf()
    #fig.set_size_inches(20,12)
    ax = sns.heatmap(data=df, annot=True, cmap='RdYlGn',linewidths=0.2, vmax=1, vmin=-1, center=0, square=True, annot_kws={'size':15})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, verticalalignment='center')
    plt.show()

def get_correlated_features_by_pearson_correlation(df_data, threshold=0.5):
    features_final = set()
    already_exists_list = []
    numeric_features = set(df_data.select_dtypes(include=[np.number]).columns)
    df_corr_orig = abs(df_data.corr(method='pearson'))
    for target_col in numeric_features:
        features_set = set()
        features_related = set()
        df_corr = df_corr_orig
        selected_columns = list(df_corr.loc[[target_col], (df_corr[target_col] >= threshold)].columns)
        if target_col in selected_columns:
            selected_columns.remove(target_col)
        df_corr = df_corr.loc[[target_col], selected_columns]
        df_corr_cols = df_corr.columns
        for i in range(len(df_corr_cols)):
            features_set.clear()
            features_related.clear()
            f_name = df_corr_cols[i]
            f_value = df_corr.iloc[0, i]
            for j in range(0, len(df_corr_cols)):
                if f_value < df_corr.iloc[0, j]:
                    f_name, f_value = df_corr_cols[j], df_corr.iloc[0, j]
                features_related.add(df_corr_cols[i])
                features_related.add(df_corr_cols[j])
            features_set.add(f_name)
        if len(features_set)<1:
            features_set.add(target_col)
        features_related.add(target_col)
        already_exists_list = [f for f in features_related if f in features_final]
        if len(already_exists_list)>0:
            features_set = already_exists_list
        for f in features_set:
            features_final.add(f)
    return sorted(list(features_final))
    
def plot_decision_boundary(model, X, y, y_predicted=None):

    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 500)
    vticks = np.linspace(bmin, bmax, 500)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    
    # make prediction with the model and reshape the output so contourf can plot it
    y_pred_enc = model.predict(ab)

    y_pred_enc = np.round(y_pred_enc)
    y_pred=np.array([np.argmax(y, axis=None, out=None) for y in y_pred_enc])
    Z = y_pred.reshape(aa.shape)

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    cntr1 = ax1.contourf(aa, bb, Z, cmap='Pastel1', alpha=0.2)
#    ax1.clabel(cntr1, inline=True, fontsize=10, use_clabeltext=True, colors='b')
    
    if y_predicted is not None:
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2', ec='silver', marker='H', s=100, alpha=0.9)
        ax1.scatter(X[:, 0], X[:, 1], c=y_predicted, cmap='Dark2', ec='silver', marker='o', s=50)
    else:
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2', ec='silver', marker='H', s=100)
    return plt

