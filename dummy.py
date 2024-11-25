import numpy as np
import pandas as pd

num_users = 600
input_days = 7
prediction_days = 14
behavior_features = 30
portrait_features = 60

np.random.seed(42)

user_ids = np.arange(1, num_users + 1)

user_data = np.random.rand(num_users, 5)

os_values = np.random.choice(["ios", "android"], num_users)
country_values = np.random.choice(["US", "KR", "JP", "IT", "FR", "DE"], num_users)
language_values = np.random.choice(["korean", "english", "japanese", "italian", "french", "german"], num_users)

behavior_data = np.random.rand(num_users, input_days, behavior_features)
portrait_data = np.random.rand(num_users, input_days, portrait_features)

y_purchase = np.array([
    np.sort(np.random.randint(1, 101, prediction_days)) for _ in range(num_users)
])

y_churn = np.zeros((num_users, prediction_days), dtype=int)

for user in range(num_users):
    last_7_days_activity = behavior_data[user, :, :].sum(axis=1)
    
    if last_7_days_activity.sum() == 0:
        y_churn[user, :] = np.random.choice([1, 0], size=prediction_days, p=[0.8, 0.2])
    else:
        y_churn[user, :] = np.random.choice([1, 0], size=prediction_days, p=[0.2, 0.8])

user_data_df = pd.DataFrame(user_data, columns=[f"user_feature_{i+1}" for i in range(user_data.shape[1])])
user_data_df.insert(0, "user_id", user_ids)
user_data_df["os"] = os_values
user_data_df["country"] = country_values
user_data_df["language"] = language_values
user_data_df.to_csv("user.csv", index=False)

behavior_list = []
for user_idx, user_id in enumerate(user_ids):
    for day in range(1, input_days + 1):
        row = list(behavior_data[user_idx, day - 1])
        behavior_list.append([user_id, day] + row)
behavior_data_df = pd.DataFrame(behavior_list, columns=["user_id", "day"] + [f"behavior_feature_{i+1}" for i in range(behavior_features)])
behavior_data_df.to_csv("behavior.csv", index=False)

portrait_list = []
for user_idx, user_id in enumerate(user_ids):
    for day in range(1, input_days + 1):
        row = list(portrait_data[user_idx, day - 1])
        portrait_list.append([user_id, day] + row)
portrait_data_df = pd.DataFrame(portrait_list, columns=["user_id", "day"] + [f"portrait_feature_{i+1}" for i in range(portrait_features)])
portrait_data_df.to_csv("portrait.csv", index=False)

y_purchase_list = []
for user_idx, user_id in enumerate(user_ids):
    for day in range(1, prediction_days + 1):
        y_purchase_list.append([user_id, day, y_purchase[user_idx, day - 1]])
y_purchase_df = pd.DataFrame(y_purchase_list, columns=["user_id", "day", "purchase"])
y_purchase_df.to_csv("purchase.csv", index=False)

y_churn_list = []
for user_idx, user_id in enumerate(user_ids):
    for day in range(1, prediction_days + 1):
        y_churn_list.append([user_id, day, y_churn[user_idx, day - 1]])
y_churn_df = pd.DataFrame(y_churn_list, columns=["user_id", "day", "churn"])
y_churn_df.to_csv("churn.csv", index=False)