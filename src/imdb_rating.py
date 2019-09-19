#%% [markdown]
# # Predicting movie's IDMB rating

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from random import randrange

#%% [markdown]
#  ## Import and process CSV
#  * Remove row which contain empty cell
#  * Reset Index for easier preprocessing step

#%%
url = ('./IMDB-Movie-Data.csv')
csv = pd.read_csv(url, sep=",")
csv = csv.dropna(axis="index", how="any")
csv = csv.sort_values(by=["Rating"], ascending=False)
csv = csv.reset_index(drop=True)
csv.head()

#%% [markdown]
#  ## Feature & target selection
#  * Select relevant feature (excluding XYZ)
#  * Feature "Description" was removed because NLP (Natural Language Processing) isn't used
#  * Feature "Metascore" was selected because Metascore usually comes out before movie release
#  * On data pre-processing step we will remove some feature

#%%
x = csv[["Genre", "Director", "Actors",
         "Runtime (Minutes)", "Revenue (Millions)", "Votes", "Metascore", "Year"]]
y = csv["Rating"]

x.head()

#%% [markdown]
#  ## Data pre-processing #1
#  * Convert feature "Genre" & "Actors" data type from string to list because a movie could have more than a genre and an actor
#  * Change feature "Year" into movie age (since release year)
#  * Rename column for convenience sake

#%%
list_genre = []
for i in range(0, x["Genre"].size):
    list_genre.append(x["Genre"][i].split(','))
x["Genre"].update(pd.Series(list_genre))

list_actors = []
for i in range(0, x["Actors"].size):
    list_actors.append(x["Actors"][i].split(','))
x["Actors"].update(pd.Series(list_actors))

# Column position, column name, column value
x.insert(3, "Age", [2019] - x["Year"])
x = x.rename(columns={"Revenue (Millions)": "Revenue"})
x = x.rename(columns={"Runtime (Minutes)": "Runtime"})
x = x.drop(columns=["Year"])

x.head()

#%% [markdown]
#  ## Data pre-processing #2
#  Perform normalization to all features which uses number/decimal. Normalization usually have range -1 to -1 or 0 to 1.

#%%
ndarray_runtime = np.array(x["Runtime"])
ndarray_metascore = np.array(x["Metascore"])
ndarray_age = np.array(x["Age"])
ndarray_votes = np.array(x["Votes"])
ndarray_revenue = np.array(x["Revenue"])

interp_runtime = np.interp(
    ndarray_runtime, (0, ndarray_runtime.max()), (0, +3))
interp_metascore = np.interp(
    ndarray_metascore, (0, ndarray_metascore.max()), (0, +5))
interp_age = np.interp(ndarray_age, (0, ndarray_age.max()), (0, +1))
interp_votes = np.interp(ndarray_votes, (0, ndarray_votes.max()), (0, +4))
interp_revenue = np.interp(
    ndarray_revenue, (0, ndarray_revenue.max()), (0, +1))

x["Runtime"].update(pd.Series(interp_runtime))
x["Metascore"].update(pd.Series(interp_metascore))
x["Age"].update(pd.Series(interp_age))
x["Votes"].update(pd.Series(interp_votes))
x["Revenue"].update(pd.Series(interp_revenue))

x.head()

#%% [markdown]
#  ## Data pre-processing #3
#  Using one-hot encode technique for feature which uses string

#%%
ohe_director = pd.get_dummies(x["Director"])

series_genre = pd.Series()
for i in range(0, x["Genre"].size):
    tmp_series = pd.Series(x["Genre"][i])
    series_genre.at[i] = tmp_series
ohe_genre = pd.get_dummies(series_genre.apply(
    pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)

series_actors = pd.Series()
for i in range(0, x["Actors"].size):
    tmp_series = pd.Series(x["Actors"][i])
    series_actors.at[i] = tmp_series
ohe_actors = pd.get_dummies(series_actors.apply(
    pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)

print(ohe_director.sample(5))
print(ohe_genre.sample(5))
print(ohe_actors.sample(5))

#%% [markdown]
#  ## Data pre-processing #4
#  Remove all row where frequency of one-hot encoded feature is too low or have extremely low correlation with target
#
#  This method is used to improve accuracy of trained model and only used with small amount of traning data

#%%
min_freq_director = 4
min_freq_genre = 6
min_freq_actors = 4

min_corr_director = 0.12
min_corr_genre = 0.11
min_corr_actors = 0.10

corr_director = []
corr_genre = []
corr_actors = []

ctr = 0
for col in ohe_director.columns:
    freq = np.sum(np.array(ohe_director[col]))
    corr = np.abs(ohe_director[col].corr(y))
    corr_director.append(corr)
    if freq < min_freq_director or corr < min_corr_director:
        ohe_director = ohe_director.drop(columns=[col])
    else:
        ctr += 1
print(
    f"Count of eligable feature 'Director' (>= {min_freq_director} && >= {min_corr_director * 100}%) : {ctr}")
print(
    f"Average correlation for feature 'Director' : {np.average(np.array(corr_director))}")


ctr = 0
for col in ohe_genre.columns:
    freq = np.sum(np.array(ohe_genre[col]))
    corr = np.abs(ohe_genre[col].corr(y))
    corr_genre.append(corr)
    if freq < min_freq_genre or corr < min_corr_genre:
        ohe_genre = ohe_genre.drop(columns=[col])
    else:
        ctr += 1
print(
    f"Count of eligable feature 'Genre' (>= {min_freq_genre} && >= {min_corr_genre * 100}%) : {ctr}")
print(
    f"Average correlation for feature 'Genre' : {np.average(np.array(corr_genre))}")

ctr = 0
for col in ohe_actors.columns:
    freq = np.sum(np.array(ohe_actors[col]))
    corr = abs(ohe_actors[col].corr(y))
    corr_actors.append(corr)
    if freq < min_freq_actors or corr < min_corr_actors:
        ohe_actors = ohe_actors.drop(columns=[col])
    else:
        ctr += 1
print(
    f"Count of eligable feature 'Actors' (>= {min_freq_actors} && >= {min_corr_actors * 100}%) : {ctr}")
print(
    f"Average correlation for feature 'Actors' : {np.average(np.array(corr_actors))}")

#%% [markdown]
#  ## Data pre-processing #5A
#  * Add filtered one-hot encoded feature into feature DataFrame
#  * See correlation between each feature & target

#%%
show_ohe_corr = True

corr = pd.concat([x, ohe_director, ohe_genre, ohe_actors, y], axis=1,
                 sort=False) if show_ohe_corr else pd.concat([x, y], axis=1, sort=False)
corr.corr(method='pearson')

#%% [markdown]
#  ## Data pre-processing #5B
#  * Remove all irrelevant features
#  * Add polynomial features degree 2

#%%
use_polynomial = True

x = x.drop(columns=["Director", "Genre", "Actors"])
x = x.drop(columns=["Age"])

if use_polynomial:
    x_poly = PolynomialFeatures(
        2, include_bias=True, interaction_only=False).fit_transform(x)
    x = pd.concat([x, pd.DataFrame(x_poly)], axis=1, sort=False)

# x = pd.concat([x, ohe_genre], axis=1, sort=False)
x = pd.concat([x, ohe_director, ohe_genre, ohe_actors], axis=1, sort=False)

print(f"Total feature : {x.shape[1]}")
x.head()

#%% [markdown]
#  ## Split training & test data
#  * Split into 70/30 due to small training data
#  * 80/20 or lower is preffered if there are more traning data
#  * Sort x_train for visualization convience

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_test = pd.concat([x_test, y_test], axis=1)
x_test = x_test.sort_values(by="Rating")
y_test = x_test.take([28], axis=1)
y_test = y_test.iloc[:, 0]
x_test = x_test.drop(columns=["Rating"])

#%% [markdown]
#  ## Training phase
#  3 layers were used with dynamic amount of node
#%%
hls = (x.shape[1], int(x.shape[1]*0.7), int(x.shape[1]*0.49))

print(f'Hidden layer/node : {hls}')

mlpr = MLPRegressor(
    hidden_layer_sizes=hls,
    activation='relu',
    solver='adam',
    alpha=0.00005,
    learning_rate='adaptive',
    learning_rate_init=0.0005,
    max_iter=5000,
    shuffle=False,
    tol=0.00005,
    momentum=0.9,
    verbose=False
)

mlpr_model = mlpr.fit(x_train, y_train)
print(f"Training iteration : {mlpr_model.n_iter_}")

#%% [markdown]
#  ## Testing phase & result
#  * Show MSE score of train phase
#  * Show MSE, R2 and variance score of test phase

#%%
mlpr_predict = mlpr.predict(x_test)

mse = mean_squared_error(y_test, mlpr_predict)
r2 = r2_score(y_test, mlpr_predict)
evs = explained_variance_score(y_test, mlpr_predict)

print(f"MSE train : {mlpr_model.loss_}")
print(f"MSE test : {mse}")
print(f"R2 score : {r2}")
print(f"Variance score : {evs}")

#%% [markdown]
# ## Training Loss Curve

#%%
plt.style.use('seaborn')
plt.plot([i for i in range(mlpr_model.n_iter_)],
         mlpr_model.loss_curve_, label='Training error')
plt.ylabel('MSE', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.title('Training Loss Curve', fontsize=18, y=1.03)
plt.legend()

#%% [markdown]
#  ## MSE score with different training size and cross-validation method

#%%

train_sizes = ((np.zeros((1, 8)) + x.shape[0]) * np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])).astype(int)

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=mlpr,
    X=x,
    y=y,
    train_sizes=train_sizes,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    verbose=0)

train_scores_mean = -train_scores.mean(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)
plt.style.use('seaborn')
plt.plot(train_sizes / x.shape[0], train_scores_mean, label='Training error')
plt.plot(train_sizes / x.shape[0],
         validation_scores_mean, label='Validation error')
plt.ylabel('MSE', fontsize=14)
plt.xlabel('Training size', fontsize=14)
plt.title('Learning curves for MLPR', fontsize=18, y=1.03)
plt.legend()

#%% [markdown]
#  ## Comparison of predicted and actual target value (table)

#%%


def percentage_diff(predict, actual):
    if predict == actual or actual == 0 or predict == 0:
        return 0
    try:
        return round((abs(predict - actual) / actual) * 100.0, 3)
    except ZeroDivisionError:
        return 0


comparison = pd.concat(
    [pd.Series(np.array(y_test)), pd.Series(mlpr_predict)], axis=1, sort=False)
comparison[2] = np.absolute(comparison[0] - comparison[1])

percent = []
for i in range(0, comparison[0].size):
    percent.append(float(percentage_diff(comparison[1][i], comparison[0][i])))
comparison[3] = percent

comparison.rename(columns={0: 'Actual Rating', 1: 'Predicted Rating',
                           2: 'Difference', 3: '% Diff.'}, inplace=True)

comparison.sample(10)


#%%
print(f"Average diff. : {np.average(comparison['Difference'])}")
print(f"Average diff. percentage : {np.average(comparison['% Diff.'])}")

#%% [markdown]
#  ## Comparison of predicted and actual target value (Curve Graph)
#  * Red point : actual rating
#  * Blue line : predicted rating

#%%
index_x = [i for i in range(0, y_test.size)]

plt.style.use('seaborn')
plt.scatter(index_x, y_test, color='red')
# plt.scatter(index_x, y_test, color='red')
plt.plot(index_x, mlpr_predict, color='blue')
# plt.scatter(index_x, mlpr_predict, color='blue')

plt.title('Rating graph')
plt.xlabel('Movie index')
plt.ylabel('Rating')

plt.show()
