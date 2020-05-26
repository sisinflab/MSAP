import pandas as pd

from util import write

train = pd.read_csv('original_trainingset.tsv', sep='\t', header=None)
test = pd.read_csv('original_testset.tsv', sep='\t', header=None)

train.columns = ['user', 'item', 'r', 't']
test.columns = ['user', 'item', 'r', 't']

data = train.copy()
data = data.append(test, ignore_index=True)

# Indexing
n_users = data['user'].nunique()
n_items = data['item'].nunique()

users_index = dict(zip(sorted(data['user'].unique()), range(0, n_users)))
items_index = dict(zip(sorted(data['item'].unique()), range(0, n_items)))

data['user'] = data['user'].map(users_index)
data['item'] = data['item'].map(items_index)

data = data.sort_values(['user', 'item', 't'])

train = pd.DataFrame(columns=data.columns)
test = pd.DataFrame(columns=data.columns)

for user_id in data['user'].unique():
    df_user = data[data['user'] == user_id]
    df_user = df_user.sort_values(by=['t'])
    train = train.append(df_user.iloc[:-1], ignore_index=True)
    test = test.append(df_user.iloc[-1], ignore_index=True)

write.save_obj(items_index, 'item_indices')
write.save_obj(users_index, 'user_indices')

train['r'] = train['r'].astype(dtype=int)
test['r'] = test['r'].astype(dtype=int)

train.to_csv('trainingset.tsv', sep='\t', header=None, index=None)
test.to_csv('testset.tsv', sep='\t', header=None, index=None)
