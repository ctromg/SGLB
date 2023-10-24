import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import scipy.optimize
from math import exp, log, sqrt
import pickle

df = pd.read_csv("covtype.csv")
df.drop(df.iloc[:, 10:df.shape[1]-1], inplace=True, axis=1)
feature, label = df.iloc[:, :-1], df.iloc[:, [-1]]
processed_data = (feature - feature.mean()) / feature.std()

# clustering
k = 50
kmeans = KMeans(n_clusters=k)
kmeans.fit(processed_data)
cluster_labels = kmeans.labels_
processed_data['Cluster'] = cluster_labels
rewards = (label == 1) * 1
processed_data['Reward'] = rewards
processed_data['Avg_Safety'] = 0
processed_data['Label'] = label
alpha = 0.4
delta = 0.01

# calculate the average reward/safety for each cluster(arm)
unique_clusters = processed_data['Cluster'].unique()
arm_reward = np.zeros((50,2))
for cluster_label in unique_clusters:
    cluster_elements = processed_data[processed_data['Cluster'] == cluster_label]
    avg_reward = cluster_elements['Reward'].mean()
    processed_data.loc[processed_data['Cluster'] == cluster_label, 'Avg_Safety'] = avg_reward
    arm_reward[cluster_label, 0] = cluster_label
    arm_reward[cluster_label, 1] = avg_reward


# find the best feasible arm
filtered_df = processed_data[processed_data['Avg_Safety'] <= alpha]
index_of_largest = filtered_df['Avg_Safety'].idxmax()
a_star = processed_data['Cluster'][index_of_largest]
opt_reward = processed_data['Avg_Safety'][index_of_largest]


T = 10000
A = np.zeros((T, 1)) # to store the arms selected
X = np.zeros((T, 10)) # to store the feature vectors
R = np.zeros((T, 1)) # reward actually in our setting R = S
Regret = np.zeros((T,1))
Safety = np.zeros((T,1))
# S = np.zeros((T, 1)) # safety
V = np.eye(10)
inv_V = np.eye(10)

A_ts = np.zeros((T, 1)) # to store the arms selected
X_ts = np.zeros((T, 10)) # to store the feature vectors
R_ts = np.zeros((T, 1)) # reward actually in our setting R = S
Regret_ts = np.zeros((T,1))
Safety_ts = np.zeros((T,1))
V_ts = np.eye(10)
inv_V_ts = np.eye(10)


def linkfun(x):
    if x < 0:
        return 1 - 1/(1+ exp(x))
    else:
        return 1/(1+exp(-x))

theta_hat = np.zeros(10)
theta_hat_ts = np.zeros(10)



for t in range(T):

    # calculate mle
    def obj(theta):
        to_sum = []
        for tau in range(t):
            to_sum.append(R[tau] - linkfun(np.inner(X[tau,:], theta)) * X[tau,:])

        return np.sum(to_sum, 0)

    def obj_ts(theta):
        to_sum = []
        for tau in range(t):
            to_sum.append(R_ts[tau] - linkfun(np.inner(X_ts[tau,:], theta)) * X_ts[tau,:])

        return np.sum(to_sum, 0)


    if t < 10:
        theta_hat = np.hstack((np.zeros(t), np.ones(1), np.zeros(9-t)))
        theta_hat_ts = np.hstack((np.zeros(t), np.ones(1), np.zeros(9-t)))
    else:
        theta_hat = scipy.optimize.root(obj, theta_hat, method='hybr').x
        theta_hat_ts = scipy.optimize.root(obj_ts, theta_hat_ts, method='hybr').x

    if np.linalg.norm(theta_hat) > 10:
        theta_hat = 10 * theta_hat / np.linalg.norm(theta_hat)

    if np.linalg.norm(theta_hat_ts) > 10:
        theta_hat_ts = 10 * theta_hat_ts / np.linalg.norm(theta_hat_ts)

    z_t = np.random.normal(size = theta_hat_ts.shape)
    theta_tilde = theta_hat_ts + sqrt(log(t+1)/(t+1)) * np.matmul(scipy.linalg.sqrtm(inv_V_ts), z_t)

    print('t=', t, 'theta_hat=',theta_hat, '\n', 'theta_hat_ts=', theta_hat_ts, '\n', 'theta_tilde=', theta_tilde)

    # generate context and pull an arm with ucb
    cluster_samples = []
    cluster_samples_ts = []
    for cluster_label in unique_clusters:
        # generate context, same for ucb and ts
        cluster_elements = processed_data[processed_data['Cluster'] == cluster_label]
        sampled_data_point = cluster_elements.sample(n=1)
        sampled_features = sampled_data_point.iloc[:, : 12].values # include reward (1 or 0) as the last feature
        cluster_label = sampled_data_point['Cluster'].values[0]


        # do the ucb-lcb and ts
        diam = sqrt(log(t+1)/(t+1)) * np.sqrt(np.dot(np.dot(sampled_features[:, :10], inv_V), sampled_features[:, :10].T))
        lcb = linkfun(np.inner(theta_hat, sampled_features[:, :10]).squeeze()) - diam
        ucb = linkfun(np.inner(theta_hat, sampled_features[:, :10]).squeeze()) + diam

        diam_ts = sqrt(log(t+1)/(t+1)) * np.sqrt(np.dot(np.dot(sampled_features[:, :10], inv_V_ts), sampled_features[:, :10].T))
        lcb_ts = linkfun(np.inner(theta_hat_ts, sampled_features[:, :10]).squeeze()) - diam_ts
        ucb_ts = linkfun(np.inner(theta_tilde, sampled_features[:, :10]).squeeze()) + np.zeros_like(diam_ts)

        # reward = sampled_features[:, 11]
        cluster_sample = np.hstack((sampled_features[:, [i for i in range(10)] + [11] ], ucb, lcb, cluster_label.reshape(-1, 1)))
        cluster_samples.append(cluster_sample)

        cluster_sample_ts = np.hstack((sampled_features[:, [i for i in range(10)] + [11] ], ucb_ts, lcb_ts, cluster_label.reshape(-1, 1)))
        cluster_samples_ts.append(cluster_sample_ts)
    context_matrix = np.vstack(cluster_samples)
    context_matrix_ts = np.vstack(cluster_samples_ts)
    # find the "best feasible arm"
    filtered_matrix = context_matrix[context_matrix[:, 12] <= alpha] #lcb
    index_of_largest = np.argmax(filtered_matrix[:, 11]) #ucb
    original_indices = np.where(context_matrix[:, 12] <= alpha)[0] #lcb
    original_index_of_largest = original_indices[index_of_largest]

    a_t = context_matrix[original_index_of_largest, 13] #arm
    x_t = context_matrix[original_index_of_largest, :10] #context
    r_t = context_matrix[original_index_of_largest, 10] #reward
    # look up the average reward of pulled arm
    r_t_expected = processed_data[processed_data['Cluster'] == a_t]['Avg_Safety'].max()

    filtered_matrix_ts = context_matrix_ts[context_matrix_ts[:, 12] <= alpha] #lcb
    index_of_largest_ts = np.argmax(filtered_matrix_ts[:, 11]) #ucb
    original_indices_ts = np.where(context_matrix_ts[:, 12] <= alpha)[0] #lcb
    original_index_of_largest_ts = original_indices_ts[index_of_largest_ts]

    a_t_ts = context_matrix_ts[original_index_of_largest_ts, 13] #arm
    x_t_ts = context_matrix_ts[original_index_of_largest_ts, :10] #context
    r_t_ts = context_matrix_ts[original_index_of_largest_ts, 10] #reward
    # look up the average reward of pulled arm
    r_t_expected_ts = processed_data[processed_data['Cluster'] == a_t_ts]['Avg_Safety'].max()

    # update values
    X[t,:] = x_t
    R[t] = r_t
    A[t] = a_t
    V = V + np.matmul(x_t.T, x_t)
    inv_V = inv_V - 1 / (1 + np.dot(x_t, np.dot(inv_V, x_t))) * np.outer(np.dot(inv_V, x_t), np.dot(inv_V, x_t))
    Regret[t] = max(0, opt_reward - r_t_expected)
    Safety[t] = max(0, r_t_expected - alpha)

    X_ts[t,:] = x_t_ts
    R_ts[t] = r_t_ts
    A_ts[t] = a_t_ts
    V_ts = V_ts + np.matmul(x_t_ts.T, x_t_ts)
    inv_V_ts = inv_V_ts - 1 / (1 + np.dot(x_t_ts, np.dot(inv_V_ts, x_t_ts))) * np.outer(np.dot(inv_V_ts, x_t_ts), np.dot(inv_V_ts, x_t_ts))
    Regret_ts[t] = max(0, opt_reward - r_t_expected_ts)
    Safety_ts[t] = max(0, r_t_expected_ts - alpha)



with open(f'SGLB.pickle', 'wb') as file:
    pickle.dump((X, R, A, Regret, Safety, X_ts, R_ts, A_ts, Regret_ts, Safety_ts), file)

# safety =