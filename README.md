```python
import os
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
```


```python
import numpy as np
import polars as pl
import pandas as pd
import plotly.graph_objects as go
pd.options.display.max_columns = None
```


```python
import lightgbm as lgb
from catboost import CatBoostRegressor
import kaggle_evaluation.mcts_inference_server
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error as mse
```


```python
class CFG:
    
    importances_path = Path('/kaggle/input/mcts-gbdt-select-200-features/importances.csv')    
    train_path = Path('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')
    batch_size = 65536

    early_stop = 1000
    n_splits = 15
    color = '#C9A9A6'
    
    lgb_w = 0.8
    lgb_p = {
        'objective': 'regression',
        'min_child_samples': 24,
        'num_iterations': 20000,
        'learning_rate': 0.07,
        'extra_trees': True,
        'reg_lambda': 0.8,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'device': 'gpu',
        'max_depth': 24,
        'max_bin': 128,
        'verbose': -1,
        'seed': 35,
        "min_data_in_bin": 1024,
    }
    
    ctb_w = 0.2
    ctb_p = {
        'loss_function': 'RMSE',
        'learning_rate': 0.07,
        'num_trees': 10000,
        'random_state': 42,
        'task_type': 'GPU',
        'reg_lambda': 0.8,
        'depth': 8
    }
```


```python
class FE:
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def drop_cols(self, df, bad_cols=None):
        
        cols = ['Id', 
                'LudRules', 
                'EnglishRules',
                'num_wins_agent1',
                'num_draws_agent1',
                'num_losses_agent1']
        
        df = df.drop([col for col in cols if col in df.columns])
        
        df.drop([col for col in df.columns if df[col].null_count() == len(df)])
        
        bad_cols = bad_cols or [col for col in df.columns if df[col].n_unique() == 1]
        
        df.drop(bad_cols)
        
        return df, bad_cols
    
    def cast_datatypes(self, df):
        
        cat_cols = ['GameRulesetName', 'agent1', 'agent2']
        
        df = df.with_columns(pl.col(cat_cols).cast(pl.Utf8))
        
        numeric_cols = list(set(df.columns) - set(cat_cols))
        
        schema = {
            col: pl.Int16 if isinstance(df.select(pl.col(col).drop_nulls().first()).item(), int) else pl.Float32
            for col in numeric_cols
        }
        
        return df.with_columns([pl.col(col).cast(dtype) for col, dtype in schema.items()])
        
    def info(self, df):
        
        print(f'Shape: {df.shape}')
        
        mem = df.estimated_size() / 1024 ** 2
        
        print(f'Memory usage: {mem:.2f}')
    
    def apply_fe(self, path):
        
        df = pl.read_csv(path, batch_size=self.batch_size)
        
        df, bad_cols = self.drop_cols(df)
        
        df = self.cast_datatypes(df)
        
        cat_cols = [col for col in df.columns if df[col].dtype == pl.String]
        
        return df, bad_cols, cat_cols
```


```python
fe = FE(CFG.batch_size)
```


```python
class MD:
    
    def __init__(self, importances_path, early_stop, n_splits, lgb_p, ctb_p, lgb_w, ctb_w, color):
        self.importances_path = importances_path
        self.early_stop = early_stop
        self.n_splits = n_splits
        self.lgb_p = lgb_p
        self.ctb_p = ctb_p
        self.lgb_w = lgb_w
        self.ctb_w = ctb_w
        self.color = color
        
    def plot_cv(self, fold_scores, title, features):
        pass
    
    def train(self, data, cat_cols, title):
        importances = pd.read_csv(self.importances_path)
        
        data[cat_cols] = data[cat_cols].astype('category')
        
        cat_cols_copy = cat_cols.copy()
        
        X = data.drop(['utility_agent1'], axis=1)
        y = data['utility_agent1']
        group = data['GameRulesetName']
        
        cv = GroupKFold(n_splits=self.n_splits)
        
        models, scores = list(), list()
        
        oof_preds = np.zeros(len(X))

        print(f'Title: {title}')
        
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y, group), 1):
            
            drop_features = importances['drop_features'].tolist()
            cat_cols = [col for col in cat_cols_copy if col not in drop_features]
            
            X_train, X_valid = X.iloc[train_index].drop(drop_features, axis=1), X.iloc[valid_index].drop(drop_features, axis=1)
            
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
            model = None
            
            if title.startswith('LightGBM'):
                
                model = lgb.LGBMRegressor(**self.lgb_p)
                
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[lgb.early_stopping(self.early_stop, verbose=0), lgb.log_evaluation(0)])
                
            elif title.startswith('CatBoost'):
                
                model = CatBoostRegressor(**self.ctb_p, verbose=0, cat_features=cat_cols)
                
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=self.early_stop, verbose=0)
            
            models.append(model)
            
            oof_preds[valid_index] = model.predict(X_valid)
            
            score = mse(y_valid, oof_preds[valid_index], squared=False)

            print(f'fold: {fold}, score: {score}')
        
            scores.append(score)
        
        print(f'average score: {sum(scores) / len(scores)}')
            
        return models, oof_preds
    
    
    def inference(self, data, cat_cols, lgb_models, ctb_models, lgb_models_oof, ctb_models_oof):

        importances = pd.read_csv(self.importances_path)

        drop_features = importances["drop_features"].tolist()

        data = data.drop(drop_features, axis=1)

        for col in cat_cols:
            data[col] = data[col].astype('category')

        data['lgb_oof_preds'] = np.mean([model.predict(data) for model in lgb_models], axis=0)

        data['ctb_oof_preds'] = np.mean([model.predict(data) for model in ctb_models], axis=0)

        lgb_preds = np.mean([model.predict(data) for model in lgb_models_oof], axis=0)
        ctb_preds = np.mean([model.predict(data) for model in ctb_models_oof], axis=0)

        return lgb_preds * self.lgb_w + ctb_preds * self.ctb_w
```


```python
md = MD(CFG.importances_path, CFG.early_stop, CFG.n_splits, CFG.lgb_p, CFG.ctb_p, CFG.lgb_w, CFG.ctb_w, CFG.color)
```


```python
def train_model():

    global bad_cols, cat_cols, lgb_models, ctb_models, lgb_models_oof, ctb_models_oof

    train, bad_cols, cat_cols = fe.apply_fe(CFG.train_path)

    train = train.to_pandas()

    lgb_models, lgb_oof_preds = md.train(train, cat_cols, 'LightGBM')
    ctb_models, ctb_oof_preds = md.train(train, cat_cols, 'CatBoost')

    train['lgb_oof_preds'] = lgb_oof_preds
    train['ctb_oof_preds'] = ctb_oof_preds

    lgb_models_oof, _ = md.train(train, cat_cols, title='LightGBM (+ OOF Preds)')
    ctb_models_oof, _ = md.train(train, cat_cols, title='CatBoost (+ OOF Preds)')
    
```


```python
counter = 0
def predict(test, submission):
    
    global counter
    
    if counter == 0:
        train_model() 
        
    counter += 1
    
    test, _ = fe.drop_cols(test, bad_cols)
    test = fe.cast_datatypes(test)
    test = test.to_pandas()
    
    return submission.with_columns(pl.Series('utility_agent1', md.inference(test, cat_cols, lgb_models, ctb_models, lgb_models_oof, ctb_models_oof)))
    
```


```python
inference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv',
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'
        )
    )
```

    Title: LightGBM


    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.
    1 warning generated.


    fold: 1, score: 0.4134299556413397
    fold: 2, score: 0.43690663265773105
    fold: 3, score: 0.3974201575579275
    fold: 4, score: 0.42218904255436146
    fold: 5, score: 0.40770190591614147
    fold: 6, score: 0.47317603045417556
    fold: 7, score: 0.42384550846491725
    fold: 8, score: 0.4693755716345275
    fold: 9, score: 0.4340338408661618
    fold: 10, score: 0.4581177806847453
    fold: 11, score: 0.4192931268223858
    fold: 12, score: 0.420825816045158
    fold: 13, score: 0.4601439334676286
    fold: 14, score: 0.43137790194098025
    fold: 15, score: 0.45830888028334527
    average score: 0.43507640566610184
    Title: CatBoost
    fold: 1, score: 0.4310429264755486
    fold: 2, score: 0.4658040281555537
    fold: 3, score: 0.447994969232344
    fold: 4, score: 0.45445686677090263
    fold: 5, score: 0.42885123736318687
    fold: 6, score: 0.5069856631198474
    fold: 7, score: 0.479319208558243
    fold: 8, score: 0.49642567298224016
    fold: 9, score: 0.4956120880535662
    fold: 10, score: 0.4636303213958491
    fold: 11, score: 0.464748139610389
    fold: 12, score: 0.4841917188023314
    fold: 13, score: 0.4707432439912538
    fold: 14, score: 0.4575442016378998
    fold: 15, score: 0.5311750808607355
    average score: 0.47190169113399266
    Title: LightGBM (+ OOF Preds)
    fold: 1, score: 0.3904785869957572
    fold: 2, score: 0.4422477329483098
    fold: 3, score: 0.39626009414298313
    fold: 4, score: 0.42108666736699646
    fold: 5, score: 0.39444182328681093
    fold: 6, score: 0.4748330360975992
    fold: 7, score: 0.4104047326759573
    fold: 8, score: 0.48193345187762926
    fold: 9, score: 0.4195455415141938
    fold: 10, score: 0.46152668326840507
    fold: 11, score: 0.4055152425729376
    fold: 12, score: 0.4038215090470482
    fold: 13, score: 0.4711254172793677
    fold: 14, score: 0.4211243220829488
    fold: 15, score: 0.45625875464939747
    average score: 0.4300402397204227
    Title: CatBoost (+ OOF Preds)
    fold: 1, score: 0.40002560545108146
    fold: 2, score: 0.4399214261852247
    fold: 3, score: 0.40180623356811157
    fold: 4, score: 0.4169060717362112
    fold: 5, score: 0.39048300295779126
    fold: 6, score: 0.4684071825362895
    fold: 7, score: 0.43694299585555
    fold: 8, score: 0.46625276125086657
    fold: 9, score: 0.4497949785619951
    fold: 10, score: 0.44700968749532194
    fold: 11, score: 0.42626143556145213
    fold: 12, score: 0.4212796129830939
    fold: 13, score: 0.4592239481411583
    fold: 14, score: 0.4234054079459681
    fold: 15, score: 0.4734613347186772
    average score: 0.43474544566325285



```python

```
