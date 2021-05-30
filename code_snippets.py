import sys
sys.path.append('D:/Study/Python')
import pankaj_python_lib as pankajimport sys
sys.path.append('D:/Study/Python')
import pankaj_python_lib as pankaj


df_loan = pd.read_csv('LoanStats Dataset_30052021.csv', low_memory=False)
df_loan.columns = [col.strip() for col in df_loan.columns]
df_loan.sort_index(axis=1, key=lambda x:x.str.strip().str.lower(), inplace=True)

df_loan.describe(include=np.number).transpose().sort_index(key=lambda x:x.str.strip().str.lower())

features_numeric = list(df_loan.select_dtypes(include=np.number).columns)
features_non_numeric = list(df_loan.select_dtypes(exclude=np.number).columns)
discrete_columns = list(df_loan.loc[:, df_loan.nunique()<=25].columns)

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


#   Plot heatmap
fig, ax = plt.subplots(figsize=(20, 15))
#ax.tick_params(labelsize=15, labelcolor='k')
ax.set_title("Heatmap for Loanstats", fontname='Bodoni 72', fontsize=30, color='b')
sns.heatmap(df_loan.corr(), ax = ax, cmap='coolwarm', annot=True, square=True, color='b')
ax.set_xticklabels(ax.get_xticklabels(), fontname='Trebuchet MS', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontname='Arial', fontsize=12, color='b')
plt.show()


def normalize_with_standard_scaler(df, col):
    std_scaler = StandardScaler()
    df[col] = std_scaler.fit_transform(df[col].values.reshape(-1, 1))
    
def normalize_with_minmax_scaler(df, col):
    minmax_scaler = MinMaxScaler()
    df[col] = minmax_scaler.fit_transform(df[col].values.reshape(-1, 1))

def find_features_by_pearson_correlation(df, target_col, method='pearson', threshold=0.07):
    numeric_cols = df.select_dtypes(include=[np.number])
    if target_col not in(numeric_cols):
        raise ValueError(f'{target_col} does not exist or is not numeric')
        
    features_set = set()
    features_related = set()
    df_corr = df.corr(method='pearson')

    df_corr = df_corr[((abs(df_corr) >= 0.5))]
    selected_columns = list(df_corr.loc[[target_col], ((abs(df_corr[target_col]) >= 0.5))].columns)
    selected_columns.remove(target_col)

    df_corr = df_corr.loc[[target_col], selected_columns]
    df_corr = abs(df_corr)
    df_corr_cols = df_corr.columns
    for i in range(len(df_corr_cols)):
        features_set.clear()
        features_related.clear()
        
        f_name = df_corr_cols[i]
        f_value = abs(df_corr.iloc[0, i])
        for j in range(i, len(df_corr_cols)):
            if df_corr_cols[i]!= df_corr_cols[j] and abs(abs(df_corr.iloc[0, i]) - abs(df_corr.iloc[0, j])) <= threshold:
                if abs(f_value) < abs(df_corr.iloc[0, j]):
                    f_name, f_value = (df_corr_cols[j], df_corr.iloc[0, j])
            features_related.add(df_corr_cols[i])
            features_related.add(df_corr_cols[j])
        features_set.add(f_name)
    if len(features_set)<1:
        features_set.add(target_col)
    features_related.add(target_col)
    return sorted(list(features_set)), sorted(list(features_related))


def find_correlated_features(df_data, numeric_features):
    features_final = set()
    for col in numeric_features:
        feature_set, feature_related = find_features_by_pearson_correlation(df_data[numeric_features], col , threshold=0.01)

        already_exists_list = [f for f in feature_related if f in features_final]

        if len(already_exists_list)>0:
            feature_set = already_exists_list

        for f in feature_set:
            features_final.add(f)
            feature_related.remove(f)

    #    for f in feature_related:
     #       features_numeric_to_check.remove(f)

    #    print()
    #    print(col, ' == ', feature_set, ' == ', feature_related)
    #print('========================================')
    correlated_features_list = sorted(list(features_final))
    return correlated_features_list


features_correlated = find_correlated_features(df_loan, features_numeric)
features_correlated


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