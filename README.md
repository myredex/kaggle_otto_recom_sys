```python
import pandas as pd
import numpy as np
import json
from scipy import sparse as sp
from tqdm.notebook import tqdm
from collections import defaultdict
import pickle
import sys
```


```python
# Create blank new collection's dicts
cnt = defaultdict(int)
clicks_cnt = defaultdict(int)
carts_cnt = defaultdict(int)
orders_cnt = defaultdict(int)

# Load json to dataframes
data = pd.read_json('train_sessions.jsonl', chunksize=50000, lines=True)

# For each dataframe (chunk)
for row in data:
    for event in tqdm(row['events']):
        
        for product in event:
            
            # add +1 to each collection
            cnt[product["aid"]] += 1
            
            if product["type"]=='orders':
                orders_cnt[product["aid"]] += 1
                
            elif product["type"]=='carts':
                carts_cnt[product["aid"]] += 1
                
            else:
                clicks_cnt[product["aid"]] += 1
```


      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/50000 [00:00<?, ?it/s]



      0%|          | 0/19403 [00:00<?, ?it/s]



```python
# Save those dicts to future
pickle.dump(cnt, open('tmp_cut/cnt_documents.pkl', "wb"))
pickle.dump(clicks_cnt, open('tmp_cut/cnt_clicks.pkl', "wb"))
pickle.dump(carts_cnt, open('tmp_cut/cnt_carts.pkl', "wb"))
pickle.dump(orders_cnt, open('tmp_cut/cnt_orders.pkl', "wb"))
```


```python
# Load data from pickles
#cnt = pd.read_pickle('tmp_cut/cnt_documents.pkl')
#cnt_clicks = pd.read_pickle('tmp_cut/cnt_clicks.pkl')
#cnt_carts = pd.read_pickle('tmp_cut/cnt_carts.pkl')
#cnt_orders = pd.read_pickle('tmp_cut/cnt_orders.pkl')
```


```python
# Create list of top products
_tmp = list(cnt.keys())
top_products = sorted(_tmp, key=lambda x: -cnt[x])

list_of_products = pd.DataFrame(top_products)
#list_of_products.to_pickle('list_of_products.pkl')
```


```python
# Create dataframe for EDA
df = pd.DataFrame(cnt.items(), columns=['aid', 'num_actions'])
```


```python
df.shape
```




    (700198, 2)




```python
df[df['num_actions']>=1].shape
```




    (700198, 2)




```python
# create new small dataframe
small_df = df.copy()# [df['num_actions']>=2]
small_df.reset_index(drop=True, inplace=True)
small_df.shape
```




    (700198, 2)



## Data preparation

I need to prepare two dicts for future counting


```python
# Create first dict for pair index = product_id
index_pid_dict = small_df['aid'].to_dict() 
```


```python
# Create second dict for pair product_id = index
small_df['ind'] = small_df.index

small_df.set_index(['aid'], inplace=True)

pid_index_dict = small_df['ind'].to_dict()
```


```python
#pid_index_dict
```

When two dicts created define functions to operate with this dicts


```python
def pid_to_idx(dicti, prod_id):
    #returns index number input List or One Id
    if type(prod_id) == int:
        return dicti[prod_id]
    return (dicti[x] for x in prod_id)

def idx_to_pid(dicti, prod_idx):
    #returns number of the product
    if type(prod_idx) == int:
        return dicti[prod_idx]
    return (dicti[x] for x in prod_idx)

# How to use this functions
#pid_to_idx(pid_index_dict, 1680086)
#idx_to_pid(index_pid_dict, 1)
```


```python
def make_coo_row(row, num_of_prod):
    # creates one row to the matrix
    idx = []
    values = []
    items = [] 

    # Read each row in json then check if product id in our shortened list of products
    items.extend([i["aid"] for i in row if i["aid"] in pid_index_dict])
    #for i in row:
    #    if i["aid"] in index_pid_dict.values():
    #        items.extend(i["aid"])
        
    n_items = len(items)

    for pid in items:

        idx.append(pid_to_idx(pid_index_dict, pid))

        # Normalisation on count of products
        values.append(1.0 / n_items)

    return sp.coo_matrix(
        (np.array(values).astype(np.float32), ([0] * len(idx), idx)), shape=(1, num_of_prod),
    )
```


```python
data2 = pd.read_json('train_sessions.jsonl', lines=True, chunksize=100000)
```


```python
# Create blank matrix
matrix_rows = []       

for chunk in tqdm(data2):
    for row in tqdm(chunk['events']):
        matrix_rows.append(make_coo_row(row, len(pid_index_dict)))

```


    0it [00:00, ?it/s]



      0%|          | 0/100000 [00:00<?, ?it/s]



      0%|          | 0/100000 [00:00<?, ?it/s]



      0%|          | 0/100000 [00:00<?, ?it/s]



      0%|          | 0/100000 [00:00<?, ?it/s]



      0%|          | 0/100000 [00:00<?, ?it/s]



      0%|          | 0/100000 [00:00<?, ?it/s]



      0%|          | 0/69403 [00:00<?, ?it/s]



```python
# Save matrix
pickle.dump(matrix_rows, open('tmp_cut/matrix_rows_not_full.pkl', "wb"))
```


```python
# Convert coo matrix to vstack matrix
train_mat = sp.vstack(matrix_rows)
```


```python
# Save matrix
pickle.dump(train_mat, open('tmp_cut/train_mat_not_full.pkl', "wb")) 
```


```python
train_mat.shape
```




    (669403, 700198)




```python
# read train_mat from pickle if needed
# train_mat = pd.read_pickle('tmp_cut/train_mat_not_full.pkl')
```

## Training model


```python
import implicit
```


```python
model = implicit.nearest_neighbours.CosineRecommender(K=30)
```


```python
# Fit the model but matrix should be used in rotated veiw / or not test it
# From time to time it unfits, just restart the cell
model.fit(train_mat)
```

    C:\Users\ttkz\anaconda3\lib\site-packages\implicit\utils.py:138: ParameterWarning: Method expects CSR input, and was passed coo_matrix instead. Converting to CSR took 0.15033864974975586 seconds
      warnings.warn(
    


      0%|          | 0/700198 [00:00<?, ?it/s]



```python
# Save model
pickle.dump(model, open('tmp_cut/als_model_cut_rotated.pkl', "wb")) 
```

## Sumbmission part and test results


```python
test = pd.read_json('test_sessions.jsonl', chunksize=100000, lines=True)
```


```python
sessions_array =  []

for chunk in tqdm(test):
    for index, row in tqdm(chunk['events'].iteritems()):
        
        # Count one row for the matrix
        one_row = make_coo_row(row, len(pid_index_dict)).tocsr()
        
        raw_recs = model.recommend(userid=0, 
                                   user_items = one_row, 
                                   N=20, 
                                   filter_already_liked_items=False, 
                                   recalculate_user=True
        )
                           
        recommended_items = list(idx_to_pid(index_pid_dict, [idx for idx in raw_recs[0]]))
        
        #recommended_items.extend(items_not_in_dict)
        
        if not recommended_items:
            recommended_items = top_products[:20]
        
        top_events = " ".join(str(item) for item in recommended_items[:20])
        
        #print(row)
        #print('raw_recs: ', raw_recs)
        #print('recommended_items', recommended_items)
        #print('top_events', top_events)
        #print('items_not_in_dict', items_not_in_dict)
        #print('items_in_dict', items_in_dict)
        
        name_clicks = str(chunk['session'][index]) + '_clicks'
        name_carts = str(chunk['session'][index]) + '_carts'
        name_orders = str(chunk['session'][index]) + '_orders'
        
        sessions_array.append([name_clicks, top_events]) 
        sessions_array.append([name_carts, top_events]) 
        sessions_array.append([name_orders, top_events])
    
```


    0it [00:00, ?it/s]



    0it [00:00, ?it/s]



    0it [00:00, ?it/s]



    0it [00:00, ?it/s]



```python
# Create dataframe
submission = pd.DataFrame(data = sessions_array, columns=['session_type', 'labels'])
submission.to_csv('predictions.csv', index=False)
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session_type</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11954372_clicks</td>
      <td>600556 820215 221435 1646920 1798482 1679435 1...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11954372_carts</td>
      <td>600556 820215 221435 1646920 1798482 1679435 1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11954372_orders</td>
      <td>600556 820215 221435 1646920 1798482 1679435 1...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11984017_clicks</td>
      <td>1463787 1360201 908157 1523205 226280 1417023 ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11984017_carts</td>
      <td>1463787 1360201 908157 1523205 226280 1417023 ...</td>
    </tr>
  </tbody>
</table>
</div>



Best score is:<br>
{'clicks': 0.43678669484662336, 'carts': 0.3737382537805713, 'orders': 0.6030793682328399, 'total': 0.5176477665585377}

On full data, score was 0.548


```python

```
