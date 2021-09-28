
import pandas as pd
import matplotlib as plt
from catboost import CatBoostRegressor
import numpy as np


train=pd.read_csv('/Users/maksimvejnbender/Downloads/raif/raif_train.csv')
test=pd.read_csv('/Users/maksimvejnbender/Downloads/raif/test.csv')


train['split']=1
test['split']=0


train.plot(kind="scatter", x="lng", y="lat", alpha=0.05)
train.plot(kind="scatter", x="lng", y="lat", alpha=0.05)

df=pd.concat([train, test]).query('price_type == 1')

df['floor']=df['floor'].fillna(1.0)
df['street'].fillna('NAN', inplace=True)
df['per_square_meter_price'].fillna(0, inplace=True)
def antipropusk(column):
    mean=df[column].mean()
    df[column]=df[column].fillna(mean)

antipropusk('reform_mean_floor_count_500')
antipropusk('reform_mean_year_building_500')
antipropusk('reform_house_population_500')
antipropusk('reform_mean_floor_count_1000')
antipropusk('reform_mean_year_building_1000')
antipropusk('reform_house_population_1000')
antipropusk('osm_city_nearest_population')

list1 = [0.001, 0.005, 0.0075, 0.001]
list2 = ['osm_amenity_points_in_', 'osm_building_points_in_', 'osm_catering_points_in_', 'osm_crossing_points_in_', 'osm_culture_points_in_',
        'osm_finance_points_in_',  'osm_offices_points_in_', 'osm_shops_points_in_']
col_names=[]
for i in range(len(list2)):
    for j in list1:
        print(list2[i] + '{}'.format(j))
        col_names.append(list2[i] + '{}'.format(j))

mean_names = []
for i in range(0,len(col_names), 4):
        print(col_names[i][4:-16] + '_mean')
        mean_names.append(col_names[i][4:-16] + '_mean')

mean_names.append('per_square_meter_price')
for i in range(0,len(col_names), 4):
        df[col_names[i][4:-16] + '_mean'] = (df[col_names[i]] * 1
                                            + df[col_names[i+1]] * 0.2
                                            + df[col_names[i+2]] * 0.1333
                                            + df[col_names[i+3]] * 0.1)/(df.shape[0]-1)

list1 = [0.005, 0.0075, 0.01]
list2 = ['osm_healthcare_points_in_', 'osm_historic_points_in_', 'osm_hotels_points_in_', 'osm_leisure_points_in_',
        'osm_train_stop_points_in_', 'osm_transport_stop_points_in_']
col_names=[]

for i in range(len(list2)):
    for j in list1:
        print(list2[i] + '{}'.format(j))
        col_names.append(list2[i] + '{}'.format(j))

mean_names = []
for i in range(0,len(col_names), 3):
        print(col_names[i][4:-16] + '_mean')
        mean_names.append(col_names[i][4:-16] + '_mean')

mean_names.append('per_square_meter_price')
for i in range(0,len(col_names), 3):
        df[col_names[i][4:-16] + '_mean'] = (df[col_names[i]] * 1
                                            + df[col_names[i+1]] * 0.67
                                            + df[col_names[i+2]] * 0.5)/(df.shape[0]-1)

df.drop(columns=['osm_amenity_points_in_0.001',
       'osm_amenity_points_in_0.005', 'osm_amenity_points_in_0.0075',
       'osm_amenity_points_in_0.01', 'osm_building_points_in_0.001',
       'osm_building_points_in_0.005', 'osm_building_points_in_0.0075',
       'osm_building_points_in_0.01', 'osm_catering_points_in_0.001',
       'osm_catering_points_in_0.005', 'osm_catering_points_in_0.0075',
       'osm_catering_points_in_0.01', 'osm_crossing_points_in_0.001',
       'osm_crossing_points_in_0.005', 'osm_crossing_points_in_0.0075',
       'osm_crossing_points_in_0.01', 'osm_culture_points_in_0.001',
       'osm_culture_points_in_0.005', 'osm_culture_points_in_0.0075',
       'osm_culture_points_in_0.01', 'osm_finance_points_in_0.001',
       'osm_finance_points_in_0.005', 'osm_finance_points_in_0.0075',
       'osm_finance_points_in_0.01', 'osm_healthcare_points_in_0.005',
       'osm_healthcare_points_in_0.0075', 'osm_healthcare_points_in_0.01',
       'osm_historic_points_in_0.005', 'osm_historic_points_in_0.0075',
       'osm_historic_points_in_0.01', 'osm_hotels_points_in_0.005',
       'osm_hotels_points_in_0.0075', 'osm_hotels_points_in_0.01',
       'osm_leisure_points_in_0.005', 'osm_leisure_points_in_0.0075',
       'osm_leisure_points_in_0.01', 'osm_offices_points_in_0.001',
       'osm_offices_points_in_0.005', 'osm_offices_points_in_0.0075',
       'osm_offices_points_in_0.01', 'osm_shops_points_in_0.001',
       'osm_shops_points_in_0.005', 'osm_shops_points_in_0.0075',
       'osm_shops_points_in_0.01',  'osm_train_stop_points_in_0.005',
       'osm_train_stop_points_in_0.0075', 'osm_train_stop_points_in_0.01', 
       'osm_transport_stop_points_in_0.005','osm_transport_stop_points_in_0.0075',
       'osm_transport_stop_points_in_0.01','reform_mean_floor_count_500', 
       'reform_mean_year_building_500', 'reform_house_population_500', 
                 'reform_count_of_houses_500','city', 
                 'lat', 'lng', 'osm_city_nearest_name', 
                 'building_mean', 'floor'], inplace=True)

df['date'] = pd.to_datetime(df['date'])

def deviation_metric_one_sample(y_true, y_pred):
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    if np.abs(deviation) <= 0.15: 
        return 0
    elif deviation <= -0.6:
        return 9.9
    elif deviation < -0.15:
        return 1.1 * (deviation / 0.15 + 1) ** 2
    elif deviation < 0.6:
        return (deviation / 0.15 - 1) ** 2
    return 9

def deviation_metric(y_true, y_pred):
    return np.array([deviation_metric_one_sample(y_true[n], y_pred[n]) for n in range(len(y_true))]).mean()

test=df.query('split==0')
train=df.query('split==1')
od=test['id']

test_features=test.drop(['id','per_square_meter_price', 'split', 'price_type'], axis=1)
train_features=train.drop(['id','per_square_meter_price', 'split', 'price_type'], axis=1)
train_target=train['per_square_meter_price']
cat_cols = ['street', 'region']

model = CatBoostRegressor(loss_function='MAE', verbose=0)
model = model.fit(train_features, train_target, cat_features=cat_cols)

pred=model.predict(test_features)
predf=pd.DataFrame(pred)
a=pd.Series(pred)

result=pd.DataFrame({'id':od, 'per_square_meter_price':a})

result.to_csv('sub1.csv', index=False)

##Для анализа полученных результатов
##загрузить вектор фактических цен за квадратный метр и назвать переменную test_target
model.get_feature_importance(prettified=True)
scores = []
metric = deviation_metric(test_target, predictions)
scores.append(metric)
print(f'Средняя метрика по бинам: {np.mean(scores):.3f}')
print(f'Отклонение метрики по бинам: {np.std(scores):.3f}')