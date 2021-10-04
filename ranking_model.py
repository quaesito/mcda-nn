import pandas as pd
import numpy as np
from skcriteria import Data, MIN, MAX
from skcriteria.madm import closeness, simple
import matplotlib.pyplot as plt
import datetime as dt
import skcriteria
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, cross_validate
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import keras.backend as K
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline

#%% Data Exploration
import pandas as pd

df = pd.read_excel(r'.\\DATA_MCDA_DNN.xlsx',header=2,\
                   convert_float=True, dtype={'a': np.float64, 'b': np.int32})

nan_counts = pd.DataFrame()
nan_counts['NaN Counts'] = df.isna().sum()
nan_counts['Percentage NaN values [%]'] = nan_counts['NaN Counts']/len(df)
nan_counts.to_csv('nan_counts.csv')

#%% Data Cleansing
df = df.dropna(axis=1, how='all')
df = df.fillna(df.mean())
df['Cloud Point'].astype(float)
df = df.drop(['Soil Conditions', 'Copper Strip Corrosion'], axis=1)
df = df.rename(columns={"Species Name (feedstocks/plants)": "Bio-diesel"})

# Merge identical feedstocksm, average values, biodiesel acronyms
merged_df = df.groupby(('Scientific Name')).mean()
merged_df = merged_df.drop(merged_df.columns[1:len(merged_df.columns)], axis=1, inplace=True)
merged_df = pd.concat([merged_df, df.groupby(('Scientific Name')).max()])
for feedstock in range(len(merged_df['Bio-diesel'])):
  if len(''.join([i[0] for i in merged_df['Bio-diesel'][feedstock].split()]).upper()) > 1:
    merged_df['Bio-diesel'][feedstock] = ''.join([i[0] for i in merged_df['Bio-diesel'][feedstock].split()]).upper()
  merged_df['Bio-diesel'][feedstock] = ''.join([i for i in merged_df['Bio-diesel'][feedstock] if i.isalpha()])

df = df.set_index('Bio-diesel')

df = merged_df

criteria_df = df[['Fuel Density','Higher heating value  (HHV)','O2',\
                  'Oxidation Stability (hour)', 'Acid value (mg KOH/g)',\
                    'Flash Point', 'Iodine value (g I/100g) (IV)', 'Kinetic Viscosity']]
# df = df.drop(['Compression ration','Fuel Density','Higher heating value  (HHV)','O2',\
#               'Oxidation Stability (hour)', 'Acid value (mg KOH/g)',\
#               'Flash Point', 'Iodine value (g I/100g) (IV)', 'Kinetic Viscosity'], axis=1)

mtx = criteria_df.values.tolist()
criteria =[MIN, MAX, MIN, MAX, MIN, MIN, MIN, MIN]

anames=[df['Bio-diesel'].tolist()]

weights_eng = [0.205, 0.045, 0.068, 0.159, 0.091, 0.182, 0.136, 0.114]
weights_env = [0.154, 0.173, 0.135,	0.115, 0.096, 0.077, 0.058, 0.192]
weights_eco = [0.192, 0.115, 0.096,	0.173, 0.058, 0.135, 0.154,	0.077]

data = Data(mtx, criteria,
            weights=weights_eco,
            anames=df['Bio-diesel'].tolist(),
            cnames=['Fuel Density','Higher heating value  (HHV)','O2', \
                    'Oxidation Stability (hour)', 'Acid value (mg KOH/g)',\
                    'Flash Point', 'Iodine value (g I/100g) (IV)', 'Kinetic Viscosity'])

data.plot.violin()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(criteria_df['Fuel Density'], criteria_df['Higher heating value  (HHV)'],\
#            criteria_df['O2'], c='r', marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

decisions = pd.DataFrame()
decisions['Bio-diesel'] = df['Bio-diesel']
df = df.drop(['Country', 'State', 'Species Type'], axis=1)
decisions = [decisions, criteria_df]
decisions = pd.concat(decisions, axis = 1,  ignore_index=False)

#%% weighted sum model
dm = simple.WeightedSum()
dec_ws = dm.decide(data)
print(dec_ws)

# Decision scores
print(dec_ws.e_)
dec_ws.e_.points
decision_rank_ws = dec_ws.rank_.tolist()
decisions['Rank WSM'] = decision_rank_ws

dec_ws.best_alternative_, data.anames[dec_ws.best_alternative_]

#%% weighted product model
dm = simple.WeightedProduct()
dm

dec_wp = dm.decide(data)
dec_wp

# Decision scores
print(dec_wp.e_)
dec_wp.e_.points

decision_rank_wp = dec_wp.rank_.tolist()
decisions['Rank WSP'] = decision_rank_wp

dec_wp.best_alternative_, data.anames[dec_wp.best_alternative_]

#%% topsis
dm = closeness.TOPSIS()
dm

dec_topsis = dm.decide(data)
dec_topsis

print(dec_topsis.e_)
print("Ideal:", dec_topsis.e_.ideal)
print("Anti-Ideal:", dec_topsis.e_.anti_ideal)
print("Closeness:", dec_topsis.e_.closeness)
dec_topsis.best_alternative_, data.anames[dec_wp.best_alternative_]

decision_topsis = dec_topsis.rank_.tolist()
decisions['Rank TOPSIS'] = decision_topsis

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_title("Sum Norm")
data.plot.violin(mnorm="sum", ax=ax1);

ax2.set_title("Vector Norm")
data.plot.violin(mnorm="vector", ax=ax2);
f.set_figwidth(15)

print("Scikit-Criteria version:", skcriteria.VERSION)
print("Running datetime:", dt.datetime.now())

#%% Average rank
decisions['Avg Rank'] = decisions[['Rank WSM', 'Rank WSP', 'Rank TOPSIS']].mean(axis=1)
decisions['Avg Rank'] = decisions['Avg Rank'].rank(ascending=True)
decisions = decisions.set_index('Bio-diesel')

print("Scikit-Criteria version:", skcriteria.VERSION)
print("Running datetime:", dt.datetime.now())

decisions.to_csv('decisions_eco.csv')

#%% Prepare data

df = df[['Fuel Density','Higher heating value  (HHV)','O2',\
                  'Oxidation Stability (hour)', 'Acid value (mg KOH/g)',\
                    'Flash Point', 'Iodine value (g I/100g) (IV)', 'Kinetic Viscosity']]

describe_df_train = df.describe().transpose()
describe_df_train.to_csv('describe_df_train.csv')

def Remove_Outlier_Indices(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    trueList = ~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR)))
    return trueList

nonOutlierList = Remove_Outlier_Indices(df)

dfSubset = df[nonOutlierList]
dfSubset = df.fillna(dfSubset.mean())
dfSubset.boxplot(rot=20)

#%% Start Accuracy Matrix
accuracy = pd.DataFrame()
accuracy['Method'] = ['R2 Score Train','R2 Score Test', 'MSE Test', 'MAE Test']
accuracy = accuracy.set_index(['Method'])

#%% Multi-variate regression

X = df.to_numpy().reshape(-1,len(df.columns))
y = decisions['Avg Rank']

lasso_eps = 0.0001
lasso_nalpha = 20
lasso_iter = 5000

degree_min = 2
degree_max = 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

RMSE = []
test_score = []
for degree in range(degree_min,degree_max):
  model = make_pipeline(PolynomialFeatures(degree, interaction_only=False),\
                        LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,\
                                normalize=True,cv=5))
  model.fit(X_train,y_train)
  test_pred = np.array(model.predict(X_test))
  RMSE.append(np.sqrt(np.sum(np.square(test_pred-y_test))))
  test_score.append(model.score(X_test,y_test))

y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)

res_mvr = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
res_mvr.to_csv('predictions_mvr.csv')

res1 = res_mvr.head(25)
res1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

accuracy['MVR'] = [r2_score(y_train, y_pred_train), r2_score(y_test, y_pred), \
                   mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)]

print('R2 Score : ', r2_score(y_test, y_pred))
print('Mean Squared Error: ',mean_squared_error(y_test, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

#%% DNN
X = dfSubset.values.reshape(-1,len(df.columns))
y = decisions['Avg Rank'].values

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X),index=decisions.index)
X_scaled = X_scaled.rename(columns={'0': 'Avg Rank'})
y_scaled = pd.DataFrame(scaler.fit_transform(y.reshape(-1,1)),index=decisions.index)
y_scaled = y_scaled.rename(columns={'0': 'Avg Rank'})

y_scaled = scaler.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, \
                                                    test_size=0.3, random_state=0)

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

model = keras.Sequential()

n_cols = X.shape[1]

model.add(layers.Dense(50, activation='relu', input_shape=(n_cols,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))

model.summary()

mc = callbacks.ModelCheckpoint('best_model.h5', monitor=r2_keras, mode='max', \
                     verbose=1, save_best_only=True)
es = callbacks.EarlyStopping(monitor='mse', mode='min', patience=1)

model.compile(optimizer='adam', loss='mse', \
              metrics=[r2_keras, 'mse','mae'])

#train model
train_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), \
                          epochs=1500, callbacks = [mc,es])

model.save('dnn_model.h5')

# show_train_history(train_history,'loss','val_loss')

train_acc = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy training : {:.3f}'.format(train_acc[1]))
scores = model.evaluate(X_test, y_test)
print('R2 Score : ', scores[1])
print('Mean Squared Error: ',scores[2]*10000)
print('Mean Absolute Error: ', scores[3]*100)

accuracy['DNN'] = [train_acc[1], scores[1], \
                   scores[2], scores[3]]

y_pred = model.predict(X_test)
y_pred = y_pred.reshape(-1,1)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

res_dnn = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

res_dnn.to_csv('predictions_dnn.csv')
res1 = res_dnn.head(25)
res1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#%% Tune parameters for MLP

def mlp_model(X, Y):

  estimator=MLPRegressor()

  param_grid = {'hidden_layer_sizes': [(50,50,50), (50,50,50), (50,100,50), (100,1)],
            'activation': ['relu','tanh','logistic'],
            'alpha': [0.001, 0.01, 0.1],
            'batch_size':['auto'],
            'learning_rate': ['constant','adaptive'],
            'solver': ['adam']}

  gsc = GridSearchCV(
      estimator,
      param_grid,
      cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

  grid_result = gsc.fit(X, y)

  best_params = grid_result.best_params_

  best_mlp = MLPRegressor(hidden_layer_sizes = best_params["hidden_layer_sizes"],
                          activation =best_params["activation"],
                          solver=best_params["solver"],
                          max_iter= 1000, n_iter_no_change = 10,
                          verbose=1)

  scoring = {'abs_error': 'neg_mean_absolute_error',
             'squared_error': 'neg_mean_squared_error',
             'r2':'r2'}

  scores = cross_validate(best_mlp, X, Y, cv=10, scoring=scoring, \
                          return_train_score=True, return_estimator = True)
  return best_mlp, scores

best_mlp, scores = mlp_model(X_scaled, y_scaled)

#%% Train Best Multi-layer Perceptron

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, \
                                                    test_size=0.3, random_state=0)

mlp = MLPRegressor(solver='adam', activation = 'relu', \
                    hidden_layer_sizes=best_mlp.hidden_layer_sizes, max_iter=100000, \
                    learning_rate='adaptive', random_state=1,\
                    batch_size=best_mlp.batch_size, alpha=0.001, \
                    early_stopping=False, n_iter_no_change=10000,\
                    verbose=1000)

print(); print(mlp)
mlp.fit(X_train, y_train)

print('Accuracy training : {:.3f}'.format(mlp.score(X_train, y_train)))
y_pred = mlp.predict(X_test)

y_pred = y_pred.reshape(-1,1)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

mlp.score(X_test, y_pred)

accuracy['MLP'] = [mlp.score(X_train, y_train), r2_score(y_test, y_pred), \
                   mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)]

pkl_filename = "mlp_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(mlp, file)

print('R2 Score : ', r2_score(y_test, y_pred))
print('Mean Squared Error: ',mean_squared_error(y_test, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

res_mlp = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
res_mlp.to_csv('predictions_mlp.csv')
res1 = res_mlp.head(25)
res1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# plt.figure(figsize=(10,10))
# sns.regplot(y_test, y_pred, fit_reg=True, scatter_kws={"s": 100})

#%% Save accuracy
accuracy.to_csv('accuracy.csv')
