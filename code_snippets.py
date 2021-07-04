Dark2, Spectral, RdYlGn, BrBG, rainbow, RdYlBu, seismic, tab20c
import sys
sys.path.append('D:/Study/Python')
import pankaj_python_lib as pankaj

set KERAS_BACKEND=theano (or CNTK, or tensorflow -- it's default)

!python -m pip install --upgrade keras --force
!python -m pip install --upgrade tensorflow-gpu --force
!python -m pip install --upgrade category_encoders --force
!python -m pip install --upgrade matplotlib
!python -m pip install --upgrade scikit-plot
!python -m pip install --upgrade graphviz
!python -m pip install --upgrade pydotplus
!python -m pip install --upgrade rbcde
#!python -m pip install --upgrade graphviz
#!python -m pip install --upgrade pydotplus
#!sudo apt-get install graphviz
!brew install graphviz

df_loan = pd.read_csv('LoanStats Dataset_30052021.csv', low_memory=False)
df_loan.columns = [col.strip() for col in df_loan.columns]
df_loan.sort_index(axis=1, key=lambda x:x.str.strip().str.lower(), inplace=True)

df_loan.describe(include=np.number).transpose().sort_index(key=lambda x:x.str.strip().str.lower())

features_numeric = list(df_loan.select_dtypes(include=np.number).columns)
features_non_numeric = list(df_loan.select_dtypes(exclude=np.number).columns)
discrete_columns = list(df_loan.loc[:, df_loan.nunique()<=25].columns)

status_dict = {i: val for i, val in enumerate(df_loan_orig['Status'].astype(str).map(lambda x:x.split("Status:")[1].strip() if x.count('Status:')>0 else x.strip()).value_counts(ascending=True).index)}
status_dict_r = {val:key for key, val in status_dict.items()}

for date_f in features_date:
    df_loan[date_f] = pd.to_datetime(df_loan[date_f], format='%Y-%m-%d')
    df_loan[date_f].fillna(pd.to_datetime(df_loan[date_f].dropna().astype(np.int64).mean()).date(), inplace=True)

import matplotlib.style as style
style.available
style.use('seaborn-notebook')


pd.set_option('display.max_columns', None)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
pd.options.display.float_format = "{:,.2f}".format

pd.options.display.max_columns=500
pd.options.display.max_rows=500
pd.options.display.max_colwidth=500
pd.options.display.max_info_columns=500
pd.options.display.max_info_rows=2000
pd.options.display.expand_frame_repr=True
pd.options.display.width=2000
plt.style.use('ggplot')


#columns having na values
df_train.columns[df_train.isna().any()].size


# %age of na values per column
na_map_per = {x:round(df_train[x].isna().mean()*100, 2) for x in df_train.columns[df_train.isna().any()]}
na_map_per = dict(sorted(list(na_map_per.items()), key=lambda x:(x[1], x[0].lower()), reverse=True))
na_map_per



# counnt of na values per column
na_map = {x:df_train[x].isna().sum() for x in df_train.columns if df_train[x].isna().any()}
na_map = dict(sorted(list(na_map.items()), key=lambda x:(x[1], x[0].lower()), reverse=True))
na_map


#np.round(df.describe(),decimals=2)
#pd.options.display.float_format = "{:,.2f}".format
pd.options.display.float_format = None

# pie chart for each column with na and non-na %age
for feature in features_with_na:
    data = df_train[['SalePrice']]
    data[feature] = np.where(df_train[feature].isna(), 'n/a', 'non n/a')
    data.groupby(feature)['SalePrice'].median().plot.pie(cmap='summer', autopct='%1.0f%%', pctdistance=0.5, labeldistance=1.2)
    plt.title(feature)
    plt.show()


# categorical features
df_train.dtypes[(df_train.dtypes=='O') | (df_train.dtypes=='category')].index.values


# numerical features
df.select_index(include=np.number)
features_numerical = [x[0] for x in df_train.dtypes.to_dict().items() if x[1].name not in ['object', 'category']]
df_train.dtypes[~((df_train.dtypes=='O') | (df_train.dtypes=='category'))].index.values

#features with year
features_with_year = [x for x in features_numerical if ('yr' in x.lower() or 'year' in x.lower())]
features_with_year

# find discrete features
features_numerical_discrete = [feature for feature in features_numerical if feature not in features_with_year and df_train[feature].unique().size<=25]
len(features_numerical_discrete), features_numerical_discrete

# plot discrete features
import matplotlib
for feature in features_numerical_discrete:
#    df_train.groupby(by=feature)['SalePrice'].median().plot.bar(color = list('rgbkymc'))
    df_train.groupby(by=feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature + ' vs SalePrice')
    plt.show()


# continuous numerical features
features_numerical_continuous = [feature for feature in features_numerical if feature not in features_numerical_discrete+['Id', 'SalePrice']]
len(features_numerical_continuous)

# categorical features with na
features_with_nan = [x for x in df_train.columns if df_train[x].isna().any() & (df_train[x].dtype.name in ['object', 'category'])]
features_with_nan

# describe
housing.describe().transpose()

housing['totalrooms'].sort_values(ascending=False)[:5]
sorted(housing['totalrooms'], reverse=True)[:5]

housing = housing.query('not (bedrooms < 3 and totalrooms > 14)').query('not (bedrooms < 3 and finishedsqft > 10000)')

samples = df_loan.groupby('Status').apply(lambda s: s.sample(min(len(s), 1)))
df_loan = df_loan.drop(samples.index.map(lambda x:x[1]))
df_loan.shape, samples.shape


# convert number into datetime
housing['lastsolddateint'] = pd.to_datetime(housing['lastsolddate'], format='%m/%d/%Y').astype('int64')
housing['lastsolddateint'] = housing['lastsolddateint']/1000000000
housing = housing[housing['lastsolddateint'].notnull()]

#   Plot bar chart without subplots
df_plot = df_new.groupby(['type', 'isFraud'])['type'].size().unstack()

fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=160)
#fig.set_dpi(150)

df_plot.plot(kind='bar', ax=ax, stacked=False,  alpha=0.9, fontsize=12, width=0.8, subplots=False, sort_columns=True)
ax.grid(alpha=0.35, color='grey', axis='y')

ax.set_title('Transaction Type Frequency', fontsize=15, color='b')
ax.set_xlabel('('+df_plot.index.name+')', fontsize=12, color='b')
ax.set_ylabel('(Frequency)', fontsize=12, color='b')

ax_min, ax_max = ax.get_yticks().min(), ax.get_yticks().max()
yticks = np.arange(ax_min, ax_max, round((ax_max-ax_min)/20)).astype(int)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks)

for c in ax.containers:
#    plt.bar_label(c, label_type='center', color='cyan', fontsize=10)
    plt.bar_label(c, label_type='edge', color='k', fontsize=10)

plt.tight_layout()
plt.show()



------------------------

def adjust_plot(ax, n_ticks = 20):
    import matplotlib
#    matplotlib.style.use('default')
    ax.set(facecolor=face_color)
    ax.grid(alpha=0.35, axis='both', color='grey')
    ax.set_xlabel(df_results.index.name, fontsize=10, color=title_color)
    ax.set_ylabel('Frequency', fontsize=10, color=title_color)
    
    yticks = ax.get_yticks()

    ax_min, ax_max = min(yticks), max(yticks)
    step = round((ax_max-ax_min)/(n_ticks - 1), 2)

    if step%10 > 0:
        yticks = np.around(np.arange(max(ax_min - step, 0), ax_max, step), 2)
    else:
        yticks = np.arange(max(ax_min - step, 0), ax_max + step, step)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, rotation=0)

    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    
    



edge_color = "k"
face_color = "whitesmoke"
color_map = 'Spectral'
title_color = 'Teal'

#edge_color = None
#face_color = None

fig, axs = plt.subplots(1, 2, figsize=(12, 8),
                        sharey=False, squeeze=False,
                        facecolor=face_color, edgecolor=edge_color, linewidth=1,
                        gridspec_kw={'wspace':0.25, 'hspace':0.25}
                       )

axs = axs.flat
#fig.tight_layout()
ax = axs[0]
df_results.plot(kind='line', ax=ax, stacked=False,  alpha=0.9, fontsize=10, subplots=False, sort_columns=True, cmap=color_map, linewidth=2 )
ax.set_title('Line Plot' , fontsize=10, color=title_color, y=-0.15)
adjust_plot(ax)

ax = axs[1]
df_results.plot(kind='bar', ax=ax, stacked=False,  alpha=0.9, fontsize=10, subplots=False, sort_columns=True, cmap=color_map)
ax.set_title('Bar Plot', fontsize=10, color=title_color, y=-0.15)
adjust_plot(ax)


#plt.subplots_adjust(wspace=0.4, hspace=0.4)
fig.suptitle('Models Comparison' , fontsize=15, color=title_color, ha='center', va='top', x=0.5, y=0.95)
plt.show()


rect = plt.Rectangle(
    # (lower-left corner), width, height
    (0.01, 0.01), 0.98, 0.98, fill=True, alpha=0.1, facecolor="lightgrey", edgecolor='k', linewidth=2, linestyle='-',
    zorder=1000, transform=fig.transFigure, figure=fig
)
rect1 = plt.Rectangle(
    # (lower-left corner), width, height
    (0.01, 0.01), 0.98, 0.98, fill=False, alpha=1, edgecolor='k', linewidth=1, linestyle='-',
    zorder=1000, transform=fig.transFigure, figure=fig
)
fig.patches.extend([rect, rect1])
------------------------
#   Plot bar chart with subplots

total_count = df_plot.nunique().count()
column_count = 2
row_count=total_count//2
fig, axs = plt.subplots(row_count, column_count, figsize=(12, 6), dpi=160)
#fig.set_dpi(150)

axs = df_plot.plot(kind='bar', ax=axs, stacked=False, alpha=0.9, fontsize=12, width=0.8, subplots=True)

for ax in axs:
    ax.grid(alpha=0.35, color='grey', axis='y')

    ax.set_title('Transaction Type Frequency', fontsize=15, color='b')
    ax.set_xlabel('('+df_plot.index.name+')', fontsize=12, color='b')
    ax.set_ylabel('(Frequency)', fontsize=12, color='b')
    ax_min, ax_max = ax.get_yticks().min(), ax.get_yticks().max()
    yticks = np.arange(ax_min, ax_max, round((ax_max-ax_min)/10)).astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    for c in ax.containers:
    #    plt.bar_label(c, label_type='center', color='cyan', fontsize=10)
        ax.bar_label(c, label_type='edge', color='k', fontsize=10)

plt.tight_layout()
plt.show()


with sns.axes_style('darkgrid'), sns.plotting_context("paper", font_scale=1.5):
    sns.jointplot(data=df_data
                 , height=8
                 , palette=plt.cm.BrBG.name )

with sns.axes_style('darkgrid'), sns.plotting_context("paper", font_scale=1.2):
    p = sns.pairplot(df_data ,
                 x_vars=['TV', 'Radio', 'Newspaper'], y_vars=y.name, 
    #             vars = X.columns, y_vars=y.name, 
    #             vars = X.columns,
                 kind='reg'
#                 , hue='Sales'
                 , height=4
                 , aspect=1
                 , palette=plt.cm.coolwarm.name
                , markers='o'
                 , plot_kws={ 
#                     'logistic':True,
#                     'robust':True,
                     'color': 'orange', 'line_kws':{'color':'b', 'lw':2, 'linestyle':'-'}, 'scatter_kws':{"s": 50}}
                )
    p.add_legend()



g = sns.PairGrid(data=df_data, palette=plt.cm.Accent.name, aspect=1, height=4)
#g = sns.PairGrid(data=df_data, hue='Sales')
g.map_upper(sns.regplot )
g.map_diag(sns.histplot, kde=True)
g.map_lower(sns.kdeplot)
g.add_legend()
plt.show()

n_rows = 2
n_cols = 2
fig, axs = plt.subplots(n_rows, n_cols, figsize=(4, 4))
for i, col in enumerate(df_data.columns):
    sns.boxplot(data=df_data, y=col, ax=axs[i//n_cols][i%n_cols], palette=plt.cm.Blues.name, linewidth=2)
plt.tight_layout()
plt.show()

#   Plot heatmap
fig, ax = plt.subplots(figsize=(20, 15))
#ax.tick_params(labelsize=15, labelcolor='k')
ax.set_title("Heatmap for Loanstats", fontname='Bodoni 72', fontsize=30, color='b')
sns.heatmap(df_loan.corr(), ax = ax, cmap='coolwarm', annot=True, square=True, color='b')
ax.set_xticklabels(ax.get_xticklabels(), fontname='Trebuchet MS', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontname='Arial', fontsize=12, color='b')
plt.show()


p = sns.FacetGrid(df_data, height = 6, aspect = 1) 
p.map(plt.scatter, 'TV', 'Sales', color = 'orange')
p.map(sns.regplot, 'TV', 'Sales', scatter = False, ci = 95, 
    fit_reg = True, color = 'blue') 
p.map(sns.histplot, 'TV', 'Sales', color = 'orange', alpha=0.2)
p.map(sns.kdeplot, 'TV', 'Sales', color = 'black', alpha=0.2)
p.map(sns.stripplot, 'TV', 'Sales', color='red')


-- DBSCAN clustering scatter plot with colors --

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.datasets import make_classification

edge_color = "k"
face_color = "whitesmoke"
color_map = 'Spectral'
title_color = 'Teal'

X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

model = DBSCAN(eps=0.3, min_samples=11)
model.fit(X)
yhat = model.labels_

clusters = sorted(np.unique(yhat))
print('Clusters: ' , len(clusters))

fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor=face_color, edgecolor=edge_color, linewidth=5,
                        gridspec_kw={'wspace':0.25, 'hspace':0.25})
ax.set(facecolor=face_color)
ax.set_title('clustering using DBSCAN')
c=[plt.cm.Dark2(x) for x in range(len(clusters))]
for i, cluster in enumerate(clusters):
	row_ix = np.where(yhat == cluster)
	print(f'cluster {cluster}.size: {row_ix[0].size}')
	label_value = 'Cluster - '+str(cluster+1) if cluster != -1 else 'Unclustered'
	c_arr = np.full(np.size(row_ix), cluster)
	print(c_arr.shape)
	ax.scatter(X[row_ix, 0], X[row_ix, 1], label=label_value, color=c[i])

plt.legend()
plt.show()

#aggregate and sorting multi index
df_data.groupby(['Cluster_id']).agg({'Estimated_income':['min', 'max'], 'recent_spends':['min', 'max']}).sort_values(by=[('Estimated_income', 'min')])


def normalize_with_standard_scaler(df, col):
    std_scaler = StandardScaler()
    df[col] = std_scaler.fit_transform(df[col].values.reshape(-1, 1))
    
def normalize_with_minmax_scaler(df, col):
    minmax_scaler = MinMaxScaler()
    df[col] = minmax_scaler.fit_transform(df[col].values.reshape(-1, 1))





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


features_correlated = get_correlated_features_by_pearson_correlation(df_loan, threshold=0.6)
len(features_correlated), len(df_loan.select_dtypes(include=[np.number]).columns)







#1 custom grid, adjusted range of values on y-axis
df_doubt_plot = df_doubt.groupby(['type', 'isFraud'])['isFraud'].size().unstack()

total_count = df_doubt_plot.nunique().count()
column_count = 2
row_count=total_count//2
fig, axs = plt.subplots(row_count, column_count, figsize=(12, 6), dpi=160)
#fig.set_dpi(150)

axs = df_doubt_plot.plot(kind='bar', ax=axs, stacked=False, alpha=0.9, fontsize=12, width=0.8, subplots=True)

for ax in axs:
    ax.grid(alpha=0.35, color='grey', axis='y')

    ax.set_title('Transaction Type Frequency', fontsize=15, color='b')
    ax.set_xlabel('('+df_doubt_plot.index.name+')', fontsize=12, color='b')
    ax.set_ylabel('(Frequency)', fontsize=12, color='b')
    ax_min, ax_max = ax.get_yticks().min(), ax.get_yticks().max()
    yticks = np.arange(ax_min, ax_max, round((ax_max-ax_min)/10)).astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    for c in ax.containers:
    #    plt.bar_label(c, label_type='center', color='cyan', fontsize=10)
        ax.bar_label(c, label_type='edge', color='k', fontsize=10, fmt='%d')

plt.tight_layout()
plt.show()

#2 

df_plot = df_loan[discrete_columns+['State']].iloc[:, :]
n_size = 5
n_col = 2
n_row = len(df_plot.columns)//n_col + len(df_plot.columns)%n_col
fig = plt.figure(figsize=(n_size*n_col, n_size*n_row), dpi=160)

for i, col in enumerate(df_plot.columns):
    ax = fig.add_subplot(n_row, n_col, i+1)
    ax.set_alpha(0.5)
    df_col = df_plot[col].value_counts()
    #df_col = df_loan.groupby([col, 'State'])['State'].size().unstack()
    #df_col.plot(kind='pie', ax=ax, fontsize=n_size*2, autopct='%.2f%%')
    df_col.plot(kind='bar', ax=ax, fontsize=n_size*2, cmap='coolwarm', alpha=0.5)
    
    #df_col.plot(kind='hist', ax=ax, fontsize=n_size*2, cmap='coolwarm', alpha=0.5)
    #df_col.plot(kind='kde',  ax=ax, fontsize=n_size*2, cmap='coolwarm', alpha=0.5, secondary_y=True)

    ax.legend(loc='upper right')
#    ax.legend_ = None
    ax.set_xlabel(col, fontsize=n_size*2)
#    ax.set_xlabel(df_col.name, fontsize=n_size*2)
    ax.set_ylabel('count of ' + col , fontsize=n_size*2)
    ax.set_title(col)
    ax.grid(alpha=0.35, color='grey', axis='y')
    
    #adjust the y ticks grid
    ax_min, ax_max = ax.get_yticks().min(), ax.get_yticks().max()
    yticks = np.arange(ax_min, ax_max, round((ax_max-ax_min)/10)).astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    
    for c in ax.containers:
    #    plt.bar_label(c, label_type='center', color='cyan', fontsize=10)
        ax.bar_label(c, label_type='edge', color='k', fontsize=n_size, fmt='%d')

#plt.tight_layout()
plt.show()


#print Decision Tree model
from sklearn import tree
print(tree.export_text(DT, feature_names=list(X.columns) ))


fig = plt.figure(figsize=(50,50))
tree.plot_tree(DT, filled=True, max_depth=None, fontsize=20, feature_names=X.columns, class_names=list(status_dict.keys()))
plt.show()


#Checking Accuracy for DecisionTreeClassifier
#train the model
DT = DecisionTreeClassifier(criterion='entropy')
DT.fit(X_train,y_train)

#make predictions
DTPredict = DT.predict(X_test)
status_keys = np.array(status_dict.keys())
#note the accuracy and ROC score
Accuracy = metrics.accuracy_score(y_test,DTPredict)
precision_score_value = metrics.precision_score(y_test, DTPredict, average='weighted')
recall_score_value = metrics.recall_score(y_test, DTPredict, average='weighted')
f1_score_value = metrics.f1_score(y_test, DTPredict, average='weighted')
hamming_loss_value = metrics.hamming_loss(y_test, DTPredict)
exact_match_ratio = np.sum(y_test.values == DTPredict)/y_test.size

print("Accuracy of the Decision Tree Classifier : ",round(Accuracy,4))
print(f'precision_score_value: {round(precision_score_value, 4)}')
print(f'recall_score_value: {round(recall_score_value, 4)}')
print(f'f1_score_value: {round(f1_score_value, 4)}')
print(f'hamming_loss_value: {round(hamming_loss_value, 4)}')
print(f'exact_match_ratio: {round(exact_match_ratio, 4)}')

skplt.metrics.plot_confusion_matrix(y_test, y_pred)

sz = 20
print(f"{'Actual':{sz}} {'Perdicted':{sz}}")
print(round(sz * 1.8) * '-')
for actual, pred in zip(sample_y.values, prediction):
    ind = "*" if actual!=pred else ""
    print(f'{status_dict_reverse[actual]:{sz}} {status_dict_reverse[pred]+ind:{sz}}')


#Taking few rows of dataframe as Samples
samples = df_loan.groupby('Status').apply(lambda s: s.sample(min(len(s), 1)))
df_loan = df_loan.drop(samples.index.map(lambda x:x[1]))


from scipy import stats
df_loan = df_loan[features_correlated]
all_cols = set(df_loan.columns)
col_l = len(all_cols)
res = np.zeros((col_l, col_l), dtype='float')
for i, col in enumerate(all_cols):
    for j, col_1 in enumerate(all_cols):
        corelation, pvalue = stats.pointbiserialr(df_loan[col], df_loan[col_1])
        res[i][j] =  round(pvalue, 4)

pv_df = pd.DataFrame(res, columns=all_cols, index=all_cols )

a = pd.DataFrame(np.where(pv_df<0.05, np.nan, pv_df), index=pv_df.columns, columns=pv_df.columns)
a = a.dropna(how='all', axis=1)
a = a.dropna(how='all', axis=0)
features_correlated = set(a.columns)
df_loan = df_loan[features_correlated]


from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
    vif = pd.Series(data=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    return vif

def calc_vif_optimized_list(X, threshold=5):
    features_list = list(X.columns)
    flag = True
    while flag:
        vif = calc_vif(X)
        max_ind = vif.argmax()
        highest_vif = vif[max_ind]
        feature_highest_vif = vif.index[max_ind]
        if highest_vif > threshold:
            features_list.remove(feature_highest_vif)
            X = X[features_list]
        else:
            flag = False
        
    return vif


target_col = 'income'
target = df[target_col]
df.drop('income', axis=1, inplace=True)
catVars = ['sex', 'overtime']
df_cat = df.drop(['sex', 'overtime'], axis=1).copy()

def plot_correlation(df, target):
    k=10
    numerical_feature_columns = list(df._get_numeric_data().columns)
    cols = df[numerical_feature_columns].corr().nlargest(k, target)[target].index
    cm = df[cols].corr
    plt.figure(figsize=(10, 6))
    return sns.heatmap(cm, annot=True, cmap='viridis')

def encode_cat_var_fit_model(
    df, df_cat, catVars, target, clf, drop_one_col, plot_corr, target_col
):
    if drop_one_col == 1:
        dummy_var = pd.get_dummies(df[catVars], drop_first=True)
    else:
        dummy_var = pd.get_dummies(df[catVars], drop_first=False)
    reg_df = pd.concat([df+cat, dummy_var], axis=1)
    temp_reg_df = pd.concat([df_cat, dummy_var, target], axis=1)
    if plot_corr==1:
        plot_out = plot_correlation(temp_reg_df, targetr_col)
    model_fit = clf.fit(reg_df, target)
    coeffs = pd.Series(model_fit.coef_)
    coef_dict = dict(zip(reg_df.columns, coeffs))
    return coef_dict

linear_reg = LinearRegression()
coeff_not_dropping_one_col = encode_cat_var_fit_model(
    df, df_cat, catVar, target, linear_reg, drop_one_col=1, plot_corr=1,
    target_col = target_col
)
print('Coefficients for Linear Regresion without dropping columns: \n')
for key, value in coeff_not_dropping_one_col.items():
    print(key, ':', round(value, 3), '\n')

X = boston['data']
Y = boston['target']
names = list(boston['feature_names'])

for i in range(0, len(names)):
    y = inp_df.loc[:, inp_df.columns == names[i]]
    x = inp_df.loc[:, inp_df.columns != names[i]]
    model = sm.OLS(y, x)
    results = model.fit()
    rsq = results.rsquared
    vif = round(1/(1-rsq), 2)
    print(f'R Square value of {names[i]} : {(round(rsq, 2))}')
    print(f'VIF value of {names[i]} : {(round(vif, 2))}')


def multicollinearity_check(df,y,threshold=5,only_final_vif=True):
    '''
    Calculates the Generalized VIF (GVIF, Fox and Monette 1992) for a data set. GVIF ** (1 / (2 * Df)) ** 2 < 5 is equivalent to VIF.
    The function assumes that categorical data are typed as 'category' or 'object' and automatically performs one-hot encoding. The function
    will work properly if the data frame has columns previously one-hot encoded from binary data, but it will not work properly if the data 
    frame has multi-nomial columns that have been previously one-hot encoded.
    Args:
        df
        y
        threshold
        only_final_vif
    
    Returns:
        pandas data frame: a data frame, indexed by factor of the GVIF, GVIF^(1/2Df), VIF^(1/2Df)^2 
        dictionary: Dictionary of column names (keys) and GVIF ** (1 / (2 * Df)) ** 2 (values)
    '''

    df_x = df.drop(y,axis=1)
    # Save categorical column names, append with prefix
    onehot_list = list(df_x.select_dtypes(include=['category', 'object', 'string']).columns)

    # Since we do not include all of the indicator variables in the model so as to avoid the dummy variable trap, one of the indicator variables is dropped
    df_1hot = pd.get_dummies(df_x, drop_first=True, dummy_na=False, prefix_sep='_')

    # Create empty df to store GVIF results
    gvif_df = pd.DataFrame(columns = ['factor','GVIF','Df','GVIF^(1/2Df)', 'VIF'])

    # Iterate over columns
    for (columnName, columnData) in df_x.iteritems():

        # Select predictor as response: if dummy encoded, select all columns for variable
        # Could all be done in the first condition, but that could result in incorrect column selection with similar column names
        if columnName in onehot_list:
            X1 = df_1hot.loc[:, df_1hot.columns.str.startswith(columnName)]
            X2 = df_1hot.loc[:, ~df_1hot.columns.str.startswith(columnName)]
        else:
            X1 = df_1hot[[columnName]].values
            X2 = df_1hot.loc[:, df_1hot.columns != columnName].values

        # Calculate gvif
        gvif = np.linalg.det(np.array(np.corrcoef(X1, rowvar=False), ndmin=2)) * np.linalg.det(np.corrcoef(X2, rowvar=False)) / np.linalg.det(np.corrcoef(np.append(X1, X2, axis=1), rowvar=False))

        gvif_12df = np.power(gvif, 1 / (2 * X1.shape[1]))
        gvif_12df_sq = gvif_12df ** 2
        DF =  X1.shape[1]
    

        # Update results df
        new_row = {'factor': columnName, 'GVIF': gvif,'Df':DF, 'GVIF^(1/2Df)': gvif_12df, 'VIF': gvif_12df_sq}
        gvif_df = gvif_df.append(new_row, ignore_index=True)

    gvif_df = gvif_df.set_index('factor')
    if only_final_vif:
        gvif_df_final = gvif_df.drop(['GVIF','Df','GVIF^(1/2Df)'],axis=1)
    else:
        gvif_df_final = gvif_df 
    gvif_filter = gvif_df.loc[gvif_df['VIF'] >= threshold]['VIF'].to_dict()
    if gvif_filter:
        for i in gvif_filter.keys():
            df_x_m = df_x.drop([i],axis=1)
    else:
        df_x_m = df_x
    df_m=pd.concat([df_x_m,df[y]],axis=1)
    return gvif_df_final,gvif_filter,df_m


df_loan.loc[:, 'Interest Rate'] = df_loan['Interest Rate'].apply(lambda v:math.ceil(v)).astype(float)




df_loan['Debt-To-Income Ratio'].apply(lambda x:math.ceil(x)).value_counts().sort_index()
df_loan['Debt-To-Income Ratio'] = pd.qcut(df_loan['Debt-To-Income Ratio'].apply(lambda x:math.ceil(x)), 10).apply((lambda x:x.left)).astype(float)
df_loan['Revolving Line Utilization'].fillna(df_loan['Revolving Line Utilization'].median(), inplace=True)
df_loan['Revolving Line Utilization'] = pd.qcut(df_loan['Revolving Line Utilization'].apply(lambda x:math.ceil(x)), 10).apply((lambda x:x.left)).astype(float)


def calculate_and_print_accuracy(clsModel, X_test, y_test, label):

    y_pred = clsModel.predict(X_test)
    #note the accuracy and ROC score
    Accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_score_value = metrics.precision_score(y_test, y_pred, average='weighted')
    recall_score_value = metrics.recall_score(y_test, y_pred, average='weighted')
    f1_score_value = metrics.f1_score(y_test, y_pred, average='weighted')
    hamming_loss_value = metrics.hamming_loss(y_test, y_pred)
    exact_match_ratio = np.sum(y_test.values == y_pred)/y_test.size
    print()
    print(f'Accuracy for: {label}')
    print(f'Accuracy: {round(Accuracy,4)}')
    print(f'precision_score_value: {round(precision_score_value, 4)}')
    print(f'recall_score_value: {round(recall_score_value, 4)}')
    print(f'f1_score_value: {round(f1_score_value, 4)}')
    print(f'hamming_loss_value: {round(hamming_loss_value, 4)}')
    print(f'exact_match_ratio: {round(exact_match_ratio, 4)}')
    print()

#make predictions
X_train, X_test, y_train, y_test = train_test_split(X, status, test_size=0.3, random_state=1)
from xgboost import XGBClassifier
clsModel = XGBClassifier()
clsModel.fit(X_train, y_train)
calculate_and_print_accuracy(clsModel, X_train, y_train, 'Training accuracy')
calculate_and_print_accuracy(clsModel, X_test, y_test, 'Testing accuracy')

sample_y=samples['Status']
samples_X = samples[features_correlated]

calculate_and_print_accuracy(clsModel, samples_X, sample_y, 'Unseen Data accuracy')


import matplotlib.pyplot as plt
x = np.power(np.arange(20), 1/2)
y = list(range(20))
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection='rectilinear')
ax.set_ybound(20)
sc = ax.scatter(x, y, c=y, cmap=plt.cm.get_cmap("Accent"), s= np.array(y)*10, ec='k', vmax=20, marker='^')
cb = fig.colorbar(sc, ax=ax, location='left', shrink=0.75, aspect=50, ticks=np.arange(0, 200, 5))
#cb.set_ticks(range(0, 21, 5))
plt.show()



------------------------------------------------------
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
---------------------------
encoded_fig = plt.figure()
ax = Axes3D(encoded_fig)
p = ax.scatter(encoded_items[:,0], encoded_items[:,1], encoded_items[:,2], c=P, marker="o", picker=True, cmap="rainbow")
plt.colorbar(p, shrink=0.5)
plt.show()

angle = 3
ani = animation.FuncAnimation(encoded_fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save('kmeans_fig.gif', writer=animation.PillowWriter(fps=12))