import tensorflow as tf
import pandas as pd 
import numpy as np
import gc
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import Sequence
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.model import EncoderIncubator

def load_data(base_dir):
    user_dir = base_dir + 'user.csv'
    behavior_dir = base_dir + 'behavior.csv'
    portrait_dir = base_dir + 'portrait.csv'
    purchase_dir = base_dir + 'purchase.csv'
    churn_dir = base_dir + 'churn.csv'
    
    user_df = pd.read_csv(user_dir)
    behavior_df = pd.read_csv(behavior_dir)
    portrait_df = pd.read_csv(portrait_dir)
    purchase_df = pd.read_csv(purchase_dir)
    churn_df = pd.read_csv(churn_dir)

    user_ids = user_df['user_id'].unique()

    train_ids, temp_ids = train_test_split(user_ids, test_size=0.2, random_state=42)
    valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    def filter_dataset(df, ids):
        return df[df['user_id'].isin(ids)].reset_index(drop=True)

    train_data = {
        "user_data": filter_dataset(user_df, train_ids),
        "behavior_data": filter_dataset(behavior_df, train_ids),
        "portrait_data": filter_dataset(portrait_df, train_ids),
        "y_purchase": filter_dataset(purchase_df, train_ids),
        "y_churn": filter_dataset(churn_df, train_ids)
    }

    valid_data = {
        "user_data": filter_dataset(user_df, valid_ids),
        "behavior_data": filter_dataset(behavior_df, valid_ids),
        "portrait_data": filter_dataset(portrait_df, valid_ids),
        "y_purchase": filter_dataset(purchase_df, valid_ids),
        "y_churn": filter_dataset(churn_df, valid_ids)
    }

    test_data = {
        "user_data": filter_dataset(user_df, test_ids),
        "behavior_data": filter_dataset(behavior_df, test_ids),
        "portrait_data": filter_dataset(portrait_df, test_ids),
        "y_purchase": filter_dataset(purchase_df, test_ids),
        "y_churn": filter_dataset(churn_df, test_ids)
    }

    return train_data, valid_data, test_data


class LabelEncoderExt(object):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit_transform(self, data_list):
        return self.label_encoder.fit_transform(list(data_list) + ['unknown_label'])[:-1]

    def transform(self, data_list):
        new_data_list = ['unknown_label' if x not in self.label_encoder.classes_ else x for x in data_list]
        return self.label_encoder.transform(new_data_list)
    
    
class DataGenerator(Sequence):

    def __init__(self,
                 data,
                 input_days,
                 target_day,
                 batch_size,
                 encoders={},
                 scalers={}
                 ):

        self.user_col = None
        self.behavior_col = None
        self.portrait_col = None

        self.user_df = data['user_data']
        self.behavior_df = data['behavior_data']
        self.portrait_df = data['portrait_data']
        self.purchase_df = data['y_purchase']
        self.churn_df = data['y_churn']
        self.input_days = input_days
        self.target_day = target_day 
        self.batch_size = batch_size
        
        self.encoders = encoders
        self.scalers = scalers

        self.sample_size = len(self.user_df)
        self.n_batches = int(np.ceil(self.sample_size / float(self.batch_size)))

        self.X, self.y_purchase, self.y_churn = self._preprocess_data()

        
    def _column_list_to_indices(self, data_type, column_list):
   
        if data_type not in ['behavior', 'portrait']:
            raise ValueError("data_type should be one of 'behavior' or 'portrait'.")

        group_mapping = [1 if int(col.split('_')[-1]) % 2 != 0 else 2 for col in column_list]
        indices = [np.where(np.array(group_mapping) == group)[0] for group in range(1, max(group_mapping) + 1)]

        return [index for index in indices if len(index) > 0]

    
    def _preprocess_data(self):
       
        user = self.user_df.drop(columns=['user_id']).values
        behavior = self.behavior_df.drop(columns=['user_id', 'day']).values
        portrait = self.portrait_df.drop(columns=['user_id', 'day']).values
        purchase = self.purchase_df.drop(columns=['user_id', 'day']).values
        churn = self.churn_df.drop(columns=['user_id', 'day']).values
        
        self.user_col = self.user_df.columns[1:]
        self.behavior_col = self.behavior_df.columns[2:]
        self.portrait_col = self.portrait_df.columns[2:]
        
        self.behavior_indices = self._column_list_to_indices('behavior', self.behavior_col)
        self.portrait_indices = self._column_list_to_indices('portrait', self.portrait_col)
        # Reshape time-series data
        behavior = behavior.reshape(-1, self.input_days, len(self.behavior_col))
        portrait = portrait.reshape(-1, self.input_days, len(self.portrait_col))
        purchase = purchase.reshape(-1, self.target_day)
        churn = churn.reshape(-1, self.target_day)

        X = {
            'user': user,
            'behavior': behavior,
            'portrait': portrait
        }
        
        print("Starting Encoding and Scaling")

        for key in X:
            scaled_or_encoded_data = []
            for i in range(X[key].shape[-1]):
                sub_key = f"{key}{i}"
                column_data = X[key][..., i]
                
                if column_data.dtype == object:
                    # Apply LabelEncoder for categorical data
                    if sub_key not in self.encoders:
                        self.encoders[sub_key] = LabelEncoderExt()
                        column_data = self.encoders[sub_key].fit_transform(column_data)
                    else:
                        column_data = self.encoders[sub_key].transform(column_data)
                else:
                    # Apply StandardScaler for numerical data
                    if sub_key not in self.scalers:
                        self.scalers[sub_key] = StandardScaler()
                        column_data = self.scalers[sub_key].fit_transform(column_data.reshape(-1, 1)).flatten()
                    else:
                        column_data = self.scalers[sub_key].transform(column_data.reshape(-1, 1)).flatten()
                
                scaled_or_encoded_data.append(column_data)
        
            scaled_or_encoded_data = np.stack(scaled_or_encoded_data, axis=-1)

            if scaled_or_encoded_data.ndim == 2 and scaled_or_encoded_data.shape[-1] == len(self.user_col):
                X[key] = scaled_or_encoded_data  # For user data
            else:
                X[key] = scaled_or_encoded_data.reshape(-1, self.input_days, len(eval(f"self.{key}_col")))

        results = (list(X.values()), purchase, churn)
        del X
        gc.collect()
        return results


    def __len__(self):
        return self.n_batches


    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.sample_size)

        X_batch = tuple([x[start_index:end_index].astype(np.float32) for x in self.X])
        
        y_purchase_batch = self.y_purchase[start_index:end_index].values.astype(np.float32)
        y_churn_batch = self.y_churn[start_index:end_index].values.astype(np.float32)
        return X_batch, y_purchase_batch, y_churn_batch
    

def pretrain_encoder(train_gen):
        
        behavior_category_indice = train_gen.behavior_indices
        portrait_category_indice = train_gen.portrait_indices
        user_input, behavior_input, portrait_input = train_gen.X
        encoding_layers = []
        for inputs, category_indice in zip([behavior_input, portrait_input],
                                           [behavior_category_indice, portrait_category_indice]):
            
            for enc_idx, feature_indice in enumerate(category_indice):
                # Validity check for feature_indice
                input_dim = inputs.shape[-1]

                if max(feature_indice) >= input_dim:
                    raise ValueError(
                    f"Invalid feature_indice: {feature_indice}. "
                    f"Maximum value {max(feature_indice)} exceeds input dimension {input_dim}."
                )
                inputs_new = tf.gather(inputs, feature_indice, axis=-1)
                recon_model = EncoderIncubator(window_length=train_gen.input_days,
                                               feature_length=len(feature_indice))
                recon_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
                hist = recon_model.fit(x=inputs_new,
                                       y=Flatten()(inputs_new),
                                       batch_size=128,
                                       epochs=30,
                                       callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                                       factor=0.2,
                                                                                       patience=3,
                                                                                       min_lr=1e-6),
                                                  tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                                   patience=4,
                                                                                   verbose=1,
                                                                                   mode='min')],
                                       verbose=0)
                print(f"encoder {enc_idx} pretrained.")
                encoding_layers.append(recon_model.encoder)
        return encoding_layers


def custom_loss(loss_fn, log_sigma):

    def loss(y_true, y_pred):
        base_loss = loss_fn(y_true, y_pred)
        weighted_loss = tf.exp(-2 * log_sigma) * base_loss + log_sigma
        return 0.5 * weighted_loss
    return loss


def train(model, lr, epochs, objective, train_gen, valid_gen):
        
        if objective == 'lifetime_spend':
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_output_purchase_mae', factor=0.2, patience=3, min_lr=1e-6),
                tf.keras.callbacks.EarlyStopping(monitor='val_output_purchase_mae', patience=6, verbose=1, mode='min')
            ]
        else :
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_output_churn_auc', factor=0.2, patience=3, min_lr=1e-6, mode='max'),
                tf.keras.callbacks.EarlyStopping(monitor='val_output_churn_auc', patience=6, verbose=1, mode='max')
            ]
        # Compile the model with custom losses for each output
        loss_output_purchase = custom_loss(
            tf.keras.losses.MeanAbsoluteError(),
            model.log_sigma_purchase
        )
        loss_output_churn = custom_loss(
            tf.keras.losses.BinaryCrossentropy(),
            model.log_sigma_churn
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss={
            'output_purchase': loss_output_purchase,
            'output_churn': loss_output_churn
            },
            metrics={
                'output_purchase': 'mae',
                'output_churn': tf.keras.metrics.AUC()
            }
        )

        hist = model.fit(
            x=train_gen.X,
            y={
                'output_purchase': train_gen.y_purchase,
                'output_churn': train_gen.y_churn
            },
            validation_data=(valid_gen.X, {
                'output_purchase': valid_gen.y_purchase,
                'output_churn': valid_gen.y_churn
            }),
            epochs=epochs,
            callbacks=callbacks
        )
        return hist


def model_scorer(y_true, y_pred, metric=None):

    if metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "r2":
        return r2_score(y_true, y_pred)
    elif metric == "mse":
        return mean_squared_error(y_true, y_pred)
    elif metric == "rmse":
        return mean_squared_error(y_true, y_pred, squared=False)
    elif metric == 'precision':
        return precision_score(y_true, y_pred)
    elif metric == 'recall':
        return recall_score(y_true, y_pred)
    elif metric == 'f1_score':
        return f1_score(y_true, y_pred)
    elif metric == 'roc_auc_score':
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0
    else:
        return 0
    

def evaluate_purchase(model, test_gen):
        # Evaluate the model
        y_true = test_gen.y_purchase
        y_pred = model.predict(test_gen.X)['output_purchase']

        y_pred = y_pred * ((y_pred > 0.5) & (y_pred < 10000))
        y_pred = np.maximum.accumulate(y_pred, axis=1)

        mae = model_scorer(y_true[:, -1], y_pred[:, -1], "mae")
        r2 = model_scorer(y_true[:, -1], y_pred[:, -1], "r2")
        rmse = model_scorer(y_true[:, -1], y_pred[:, -1], "rmse")
        
        print(f"Purchase Prediction Result: MAE {mae:.4f} / R2 {r2:.4f} / RMSE {rmse:.4f}")


def evaluate_churn(model, test_gen):
        # Evaluate the model
        y_true = test_gen.y_churn
        y_pred_roc = model.predict(test_gen.X)['output_churn']
        y_pred = (y_pred_roc > 0.5).astype(int)

        precision = model_scorer(y_true[:, -1], y_pred[:, -1], "precision")
        recall = model_scorer(y_true[:, -1], y_pred[:, -1], "recall")
        f1_score = model_scorer(y_true[:, -1], y_pred[:, -1], "f1_score")
        roc_auc_score = model_scorer(y_true[:, -1], y_pred_roc[:, -1], "roc_auc_score")
        
        print(f"Churn Prediction Result:  Precision {precision:.4f} / Recall {recall:.4f} / F1 Score {f1_score:.4f} / ROC AUC {roc_auc_score:.4f}")