from autots import AutoTS
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import re
import streamlit as st
 
 
def autots_model_automate(df,date_column,target,join_col_name,forecast_length,frequency,autots_config):
 
    model = AutoTS(
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=autots_config.get("prediction_interval",0.9),
            ensemble=autots_config.get("ensemble",'auto'),
            model_list=autots_config.get("model_list",'superfast'),
            transformer_list=autots_config.get("transformer_list",'fast'),
            # drop_most_recent=5,  # Keep this for accuracy model
            max_generations=autots_config.get("max_generations",1),
            num_validations=autots_config.get("num_validations",1),
            n_jobs=5,
            verbose=False
        )
    # model = model.fit(df, date_col=date_column, value_col=target,id_col=join_col_name)
    # forecast = model.predict().forecast
    # best_scores = model.score_breakdown[model.score_breakdown.index == model.best_model_id]
    # SMAPE_Score=best_scores.mean().get('smape')
    # mape_Score=(SMAPE_Score / 2) * (1 + (SMAPE_Score / 100))
 
    # return forecast,mape_Score
    try:
        model = model.fit(df, date_col=date_column, value_col=target, id_col=join_col_name)
        forecast = model.predict().forecast
        best_scores = model.score_breakdown[model.score_breakdown.index == model.best_model_id]
        SMAPE_Score = best_scores.mean().get('smape')
        mape_Score = (SMAPE_Score / 2) * (1 + (SMAPE_Score / 100))
 
        return forecast, mape_Score
 
    except KeyError as e:
        if 'TransformationParameters' in str(e):
            print("Error: Missing 'TransformationParameters' in the template.")
        else:
            print(f"Unexpected KeyError: {e}")
 
    except Exception as e:
        print(f"An error occurred: {e}")
 
    return None, None
 
def auto_robust_parse_date(val):
    """
    Try to parse a value as a date.
    - If the value is an 8-digit number, assume YYYYMMDD.
    - Otherwise, require that the string contains at least one typical date delimiter ('-', '/', or '.'),
      and does NOT contain letters mixed with numbers.
    - If these conditions are not met, return NaT.
    Returns a datetime if parsing succeeds and the year is plausible; otherwise returns pd.NaT.
    """
    s = str(val).strip()
    if not s:
        return pd.NaT
 
    # If the string is all digits and exactly 8 characters, assume YYYYMMDD.
    if s.isdigit() and len(s) == 8:
        try:
            dt = datetime.strptime(s, '%Y%m%d')
            if 1900 <= dt.year <= 2100:
                return dt
        except ValueError:
            return pd.NaT
 
    # If it contains any letters, do not attempt parsing.
    if re.search(r"[A-Za-z]", s):
        return pd.NaT
 
    # For non-digit-only strings, require a delimiter (-, /, .).
    if not any(delim in s for delim in ['-', '/', '.']):
        return pd.NaT
 
    try:
        dt = parse(s, fuzzy=False)  # Set fuzzy=False to avoid extracting parts of strings
        if 1900 <= dt.year <= 2100:
            return dt
    except Exception:
        return pd.NaT
 
    return pd.NaT
 
# -----------------------------------------------------------
# Function to Auto-detect Multiple Datetime Columns
# -----------------------------------------------------------
def auto_detect_datetime_columns(df, selected_cols, threshold=0.8, exclude_non_date_text=True):
    """
    Detect all columns in selected_cols that have a high fraction of datetime values.
    
    Parameters:
    - df: Pandas DataFrame
    - selected_cols: List of columns to check
    - threshold: Minimum fraction of valid datetime values to classify as datetime (default 80%)
    - exclude_non_date_text: Whether to exclude columns with alphanumeric non-date patterns.
 
    Returns:
    - List of column names that qualify as datetime columns
    - Dictionary of columns with their datetime ratio
    """
    date_cols = []
    datetime_ratios = {}
 
    for col in selected_cols:
        # Only consider columns that are object or mixed type.
        if not pd.api.types.is_object_dtype(df[col]):
            continue
        
        # Exclude columns with non-date-like alphanumeric values
        if exclude_non_date_text:
            alphanumeric_ratio = df[col].astype(str).str.contains(r"[A-Za-z].*\d|\d.*[A-Za-z]").mean()
            if alphanumeric_ratio > 0.5:  # If more than 50% contain mixed text/numbers, ignore
                continue
 
        parsed = df[col].apply(auto_robust_parse_date)
        ratio = parsed.notnull().mean()  # Fraction of valid datetime parses.
 
        # Store results
        datetime_ratios[col] = ratio
 
        if ratio >= threshold:
            date_cols.append(col)
 
    return date_cols, datetime_ratios
 
def autots_run_pipeline(df, selected_feature, target, forecast_length, frequency,autots_config):
    datetime_column = None
    valid_datetime_columns = []
    join_on_symbol = '__Jos__'
 
    # for feature in selected_feature[:]:  # Iterate over a copy
    #     if feature in df.columns:
    #         # ✅ Ensure only object or datetime columns are considered
    #         if not pd.api.types.is_object_dtype(df[feature]) and not pd.api.types.is_datetime64_any_dtype(df[feature]):
    #             continue  # Skip non-date columns (like latitude)
 
    #         try:
    #             converted = pd.to_datetime(df[feature], errors="coerce", infer_datetime_format=True)
                
    #             # ✅ Only store it if all values are valid datetime (no NaT values)
    #             if converted.notna().all():  
    #                 valid_datetime_columns.append(feature)
 
    #                 # ✅ Select the first valid datetime column
    #                 if datetime_column is None:
    #                     df[feature] = converted
    #                     datetime_column = feature
 
    #         except Exception as e:
    #             print(f"Error converting {feature}: {e}")
    #             continue  # Skip to the next feature if there's an error
 
    # # ✅ Remove all valid datetime columns from selected_feature
 
    valid_datetime_columns, best_ratio = auto_detect_datetime_columns(df, selected_feature)
    selected_feature = [col for col in selected_feature if col not in valid_datetime_columns]
 
    print(f"Valid datetime columns: {valid_datetime_columns}")
    print(f"Selected feature: {selected_feature}")
    # print(datetime_column)
    
    date_column = valid_datetime_columns[0]
    features = selected_feature
    target = target
 
    print(date_column, target, features)
 
    join_col_name = join_on_symbol.join(features)  # Join with '__|__'
    print(join_col_name,'join_col_name-=========================================================---------------------')
    df[join_col_name] = df[features].astype(str).agg(join_on_symbol.join, axis=1)
 
    df = df[[date_column, target, join_col_name]]
    print(df.head())
 
    df[date_column]=pd.to_datetime(df[date_column], errors="coerce", infer_datetime_format=True)
    try:
        forecasted_df,smape_score=autots_model_automate(df,date_column,target,join_col_name,forecast_length,frequency,autots_config)
 
        forecast_df = forecasted_df.reset_index().rename(columns={"index":date_column})
        df_melted = forecast_df.melt(id_vars=[date_column], var_name=join_col_name, value_name=target)
 
        # df_melted=df_melted[[date_column,join_col_name,target]]
        original_col=join_col_name.split(join_on_symbol)
 
        
    
        df_melted[original_col] = df_melted[join_col_name].str.split(join_on_symbol, expand=True)
        # print(split_df.head())
        if len(selected_feature)>1:
            df_melted.drop(columns=[join_col_name], inplace=True)
 
        print(df_melted.head())
        forecasted_df=df_melted[[date_column]+original_col+[target]]
 
        return forecasted_df,smape_score
    
    except Exception as e:
        return None,None
 
 
 
 





from datetime import datetime

import numpy as np
import pandas as pd
import re
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Optional Bayesian tuner
try:
    from skopt import BayesSearchCV
except ImportError:
    BayesSearchCV = None




class ForecastPipeline:
    def __init__(self, xgb_params=None, fine_tune_method="None", period_option="Daily", forecast_value=10):
        # If no XGBoost parameters are provided, use default values.
        if xgb_params is None:
            xgb_params = {
                "base_score": 0.5,
                "booster": "gbtree",
                "objective": "reg:squarederror",
                "n_estimators": 500,
                "learning_rate": 0.01,
                "max_depth": 5,
                "min_child_weight": 5,
                "reg_lambda": 10,
                "reg_alpha": 5
            }
        self.xgb_params = xgb_params
        self.fine_tune_method = fine_tune_method
        self.period_option = period_option
        self.forecast_value = forecast_value
 
    # ------------------------------
    # Utility Functions
    # ------------------------------
    @staticmethod
    def robust_parse_date(val):
        """
        Try to parse a value as a date.
        - If the value is an 8-digit number, assume YYYYMMDD.
        - Otherwise, require that the string contains at least one typical date delimiter ('-', '/', or '.'),
        and does NOT contain letters mixed with numbers.
        - If these conditions are not met, return NaT.
        Returns a datetime if parsing succeeds and the year is plausible; otherwise returns pd.NaT.
        """
        s = str(val).strip()
        if not s:
            return pd.NaT
 
        # If the string is all digits and exactly 8 characters, assume YYYYMMDD.
        if s.isdigit() and len(s) == 8:
            try:
                dt = datetime.strptime(s, '%Y%m%d')
                if 1900 <= dt.year <= 2100:
                    return dt
            except ValueError:
                return pd.NaT
 
        # If it contains any letters, do not attempt parsing.
        if re.search(r"[A-Za-z]", s):
            return pd.NaT
 
        # For non-digit-only strings, require a delimiter (-, /, .).
        if not any(delim in s for delim in ['-', '/', '.']):
            return pd.NaT
 
        try:
            dt = parse(s, fuzzy=False)  # Set fuzzy=False to avoid extracting parts of strings
            if 1900 <= dt.year <= 2100:
                return dt
        except Exception:
            return pd.NaT
 
        return pd.NaT
 
    @staticmethod
    def detect_datetime_column(df, selected_cols, threshold=0.8,exclude_non_date_text=True):
        best_col = None
        best_ratio = 0
        for col in selected_cols:
            # Exclude columns with non-date-like alphanumeric values
            if exclude_non_date_text:
                alphanumeric_ratio = df[col].astype(str).str.contains(r"[A-Za-z].*\d|\d.*[A-Za-z]").mean()
                if alphanumeric_ratio > 0.5:  # If more than 50% contain mixed text/numbers, ignore
                    continue
 
            if not pd.api.types.is_object_dtype(df[col]):
                continue
            parsed = df[col].apply(ForecastPipeline.robust_parse_date)
            ratio = parsed.notnull().mean()
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = col
        return (best_col, best_ratio) if best_ratio >= threshold else (None, best_ratio)
 
    @staticmethod
    def add_time_features(df, datetime_col):
        df['year'] = df[datetime_col].dt.year
        df['month'] = df[datetime_col].dt.month
        df['day'] = df[datetime_col].dt.day
        df['weekday'] = df[datetime_col].dt.weekday
        df['hour'] = df[datetime_col].dt.hour
        df['minute'] = df[datetime_col].dt.minute
        df['second'] = df[datetime_col].dt.second
        df['week_number'] = df[datetime_col].dt.isocalendar().week.astype(int)
        df['is_month_start'] = df[datetime_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[datetime_col].dt.is_month_end.astype(int)
        df['quarter'] = df[datetime_col].dt.quarter
        return df
 
    @staticmethod
    def create_time_features(df, datetime_col, target_col, default_lags=[1, 2], default_rolling=[3, 5]):
        df = ForecastPipeline.add_time_features(df, datetime_col)
        for lag in default_lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        for window in default_rolling:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        return df
 

    #         raise ValueError("No forecast results generated for any group.")
    def run(self, df, feature_cols, target_col):
        # 1. Detect the datetime column.
        datetime_col, best_ratio = self.detect_datetime_column(df, feature_cols)
        if datetime_col is None:
            raise ValueError("Could not detect a datetime column among the selected features.")
 
        # Compute the global maximum date available in the file.
        global_max_date = pd.to_datetime(df[datetime_col], errors="coerce").max()
 
        # 2. Define grouping features: only non-numeric (excluding the datetime column).
        group_features = [col for col in feature_cols if col != datetime_col and not pd.api.types.is_numeric_dtype(df[col])]
        
        # 3. Define additional (input) features: numeric ones and any extra non‑grouping columns.
        additional_features = [col for col in feature_cols if col != datetime_col and col not in group_features]
 
        # 4. Create composite group id if grouping features exist.
        if group_features:
            join_on_symbol = '__Jos__'
            group_id_col = join_on_symbol.join(group_features)
            df[group_id_col] = df[group_features].astype(str).agg(join_on_symbol.join, axis=1)
            unique_groups = df[group_id_col].unique()
        else:
            unique_groups = [None]
            group_id_col = None
 
        forecast_results_list = []
        mape_list = []  # Collect MAPE for each group
 
        # 5. Loop over each group.
        for grp in unique_groups:
            if grp is not None:
                df_grp = df[df[group_id_col] == grp].copy()
            else:
                df_grp = df.copy()
 
            # Ensure the datetime column is parsed and sorted.
            df_grp[datetime_col] = df_grp[datetime_col].apply(self.robust_parse_date)
            if df_grp[datetime_col].isnull().any():
                continue  # Skip groups with date conversion issues.
            df_grp.sort_values(by=datetime_col, inplace=True)
 
            # 6. Process additional features.
            X_features = df_grp[additional_features].copy() if additional_features else pd.DataFrame()
            cat_cols = X_features.select_dtypes(include=['object']).columns.tolist()
            num_cols = [col for col in X_features.columns if col not in cat_cols]
            
            # Label encode categorical features.
            for col in cat_cols:
                le = LabelEncoder()
                X_features[col] = le.fit_transform(X_features[col])
            
            # Scale numeric columns.
            if num_cols:
                scaler = StandardScaler()
                X_features_scaled_num = pd.DataFrame(scaler.fit_transform(X_features[num_cols]),
                                                    columns=num_cols, index=X_features.index)
            else:
                X_features_scaled_num = pd.DataFrame(index=X_features.index)
            
            X_features_scaled_cat = X_features[cat_cols].copy() if cat_cols else pd.DataFrame(index=X_features.index)
            
            if not X_features_scaled_cat.empty and not X_features_scaled_num.empty:
                X_features_processed = pd.concat([X_features_scaled_cat, X_features_scaled_num], axis=1)
            elif not X_features_scaled_cat.empty:
                X_features_processed = X_features_scaled_cat.copy()
            else:
                X_features_processed = X_features_scaled_num.copy()
 
            # 7. Combine datetime, processed features, and target.
            df_proc = pd.concat([df_grp[[datetime_col]], X_features_processed, df_grp[[target_col]]], axis=1)
            df_proc = self.create_time_features(df_proc, datetime_col, target_col)
            df_proc.dropna(inplace=True)
 
            # Check if there is sufficient data.
            if len(df_proc) < 10:
                continue
 
            X_full = df_proc.drop(columns=[datetime_col, target_col])
            y_full = df_proc[target_col]
 
            # 8. Train-test split and train the XGBoost model.
            X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
            model = xgb.XGBRegressor(**self.xgb_params)
            search = None
            pg = {"max_depth":[3,5,7],"learning_rate":[0.01,0.05,0.1],"n_estimators":[100,300,500],"min_child_weight":[1,3,5],"reg_lambda":[0,10,20],"reg_alpha":[0,5,10]}
            if self.fine_tune_method=="Grid Search": search = GridSearchCV(model, pg, cv=3, n_jobs=-1, verbose=1)
            elif self.fine_tune_method=="Random Search": search = RandomizedSearchCV(model, pg, cv=3, n_iter=10, n_jobs=-1, verbose=1, random_state=42)
            elif self.fine_tune_method=="Bayesian Optimization" and BayesSearchCV: search = BayesSearchCV(model, pg, cv=3, n_iter=15, n_jobs=-1, verbose=1, random_state=42)
            if search: search.fit(X_train, y_train); model=search.best_estimator_
            else: model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test)], verbose=True)
            # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
 
            # Compute MAPE on the training data.
            preds_train = model.predict(X_train)
            group_mape = mean_absolute_percentage_error(y_train, preds_train)
            mape_list.append(group_mape)
 
            # 9. Recursive Forecasting using the global max date.
            # Use global_max_date for consistent forecasting across groups.
            start_date = global_max_date
            freq = {'Hourly': 'H', 'Daily': 'D', '5 Minutes': '5T', 'weekly': 'W', 'Monthly': 'M', 'Yearly': 'Y'}.get(self.period_option, 'H')
            future_dates = pd.date_range(start=start_date, periods=self.forecast_value+1, freq=freq)[1:]
 
            recursive_preds = []
            # Use the last row's features as a starting point.
            last_row = df_proc.iloc[-1]
            current_features = last_row.drop([datetime_col, target_col]).copy()
 
            # Forecast exogenous features with ARIMA.
            exog_feats = [col for col in X_full.columns if col not in ['lag_1', 'lag_2']]
            exog_forecasts = {}
            for feat in exog_feats:
                try:
                    series = df_proc[feat].dropna()
                    arima_model = ARIMA(series, order=(1,1,1)).fit()
                    forecast_series = arima_model.forecast(steps=self.forecast_value)
                    exog_forecasts[feat] = forecast_series.values
                except Exception as ex:
                    exog_forecasts[feat] = np.repeat(df_proc[feat].iloc[-1], self.forecast_value)
 
            step = 0
            for dt in future_dates:
                # st.write(current_features)
                next_pred = model.predict(current_features.values.reshape(1, -1))[0]
                recursive_preds.append(next_pred)
                new_features = current_features.copy()
                if 'lag_1' in new_features.index and 'lag_2' in new_features.index:
                    new_features['lag_2'] = current_features['lag_1']
                    new_features['lag_1'] = next_pred
                for feat in exog_feats:
                    new_features[feat] = exog_forecasts[feat][step]
                current_features = new_features.copy()
                step += 1
 
            # 10. Build forecast DataFrame for this group.
            future_df = pd.DataFrame({datetime_col: future_dates, "forecast": recursive_preds})
            
            # Include grouping columns in the output.
            if group_features:
                for gf in group_features:
                    future_df[gf] = df_grp[gf].iloc[0]
            
            # Also include additional (numeric) input columns (using last observed values).
            if additional_features:
                for col in additional_features:
                    future_df[col] = df_grp[col].iloc[-1]
            
            # Add the MAPE for this group to each row of its forecast.
            # future_df["MAPE"] = group_mape
 
            forecast_results_list.append(future_df)
 
        # 11. Combine all group forecasts.
        if forecast_results_list:
            final_forecast_df = pd.concat(forecast_results_list, axis=0)
            overall_mape = sum(mape_list) / len(mape_list) if mape_list else None
            return final_forecast_df, group_mape
        else:
            raise ValueError("No forecast results generated for any group.")
        



        

# class ForecastPipeline:
#     def __init__(
#         self,
#         xgb_params=None,
#         fine_tune_method="None",
#         period_option="Daily",
#         forecast_value=10
#     ):
#         if xgb_params is None:
#             xgb_params = {
#                 "base_score": 0.5,
#                 "booster": "gbtree",
#                 "objective": "reg:squarederror",
#                 "n_estimators": 500,
#                 "learning_rate": 0.01,
#                 "max_depth": 5,
#                 "min_child_weight": 5,
#                 "reg_lambda": 10,
#                 "reg_alpha": 5
#             }
#         self.xgb_params = xgb_params
#         self.fine_tune_method = fine_tune_method
#         self.period_option = period_option
#         self.forecast_value = forecast_value
#         self.model = None
#         self.scaler = StandardScaler()
#         self.encoders = {}
#         self.one_hot_encoded_cols = []
#         self.numerical_cols_scaled = []
#         self.features = []

#     @staticmethod
#     def robust_parse_date(val):
#         s = str(val).strip()
#         if not s:
#             return pd.NaT
#         if s.isdigit() and len(s) == 8:
#             try:
#                 dt = datetime.strptime(s, "%Y%m%d")
#                 if 1900 <= dt.year <= 2100:
#                     return pd.Timestamp(dt)
#             except ValueError:
#                 return pd.NaT
#         if re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", s):
#             return pd.NaT
#         if not any(d in s for d in ["-", "/", "."]):
#             return pd.NaT
#         try:
#             dt = parse(s, fuzzy=False)
#             if 1900 <= dt.year <= 2100:
#                 return pd.Timestamp(dt)
#         except Exception:
#             return pd.NaT
#         return pd.NaT

#     @staticmethod
#     def detect_datetime_column(df, selected_cols, threshold=0.8, exclude_non_date_text=True):
#         for col in selected_cols:
#             if pd.api.types.is_datetime64_any_dtype(df[col]):
#                 return col, 1.0
#         best_col, best_ratio = None, 0.0
#         for col in selected_cols:
#             if exclude_non_date_text:
#                 mixed = df[col].astype(str).str.contains(r"[A-Za-z].*\d|\d.*[A-Za-z]").mean()
#                 if mixed > 0.5:
#                     continue
#             parsed = df[col].apply(ForecastPipeline.robust_parse_date)
#             ratio = parsed.notna().mean()
#             if ratio > best_ratio:
#                 best_ratio, best_col = ratio, col
#         return (best_col, best_ratio) if best_ratio >= threshold else (None, best_ratio)

#     @staticmethod
#     def add_time_features(df, datetime_col):
#         dt = df[datetime_col]
#         df['year'] = dt.dt.year
#         df['month'] = dt.dt.month
#         df['day'] = dt.dt.day
#         df['weekday'] = dt.dt.weekday
#         df['hour'] = dt.dt.hour
#         df['minute'] = dt.dt.minute
#         df['second'] = dt.dt.second
#         df['week_number'] = dt.dt.isocalendar().week.astype(int)
#         df['is_month_start'] = dt.dt.is_month_start.astype(int)
#         df['is_month_end'] = dt.dt.is_month_end.astype(int)
#         df['quarter'] = dt.dt.quarter
#         return df

#     @staticmethod
#     def create_time_features(df, datetime_col, target_col, default_lags=[1, 2], default_rolling=[3, 5]):
#         df = df.copy()
#         df = ForecastPipeline.add_time_features(df, datetime_col)
#         for lag in default_lags:
#             df[f'lag_{lag}'] = df[target_col].shift(lag)
#         for window in default_rolling:
#             df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
#         return df

#     def preprocess(self, df, features, target_col):
#         df = df.dropna(subset=[target_col])
#         categorical_cols = [col for col in features if df[col].dtype == 'object' or df[col].dtype.name == 'category']
#         numerical_cols = [col for col in features if col not in categorical_cols]

#         # Label Encode
#         for col in categorical_cols:
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col])
#             self.encoders[col] = le

#         # One-Hot Encode
#         df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dummy_na=False)
#         self.one_hot_encoded_cols = [col for col in df.columns if any(cat_col in col for cat_col in categorical_cols)]
#         print("✅ One-Hot Encoded Columns (Training):", self.one_hot_encoded_cols)

#         # Scale numerical features
#         if numerical_cols:
#             self.numerical_cols_scaled = numerical_cols
#             df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
#         else:
#             self.numerical_cols_scaled = []

#         self.all_processed_features = self.numerical_cols_scaled + self.one_hot_encoded_cols
#         self.final_training_columns = self.all_processed_features  # Store the final column names
#         return df[self.all_processed_features + [target_col]]

#     def train(self, df, datetime_col, target_col, selected_features, group_cols=None):
#         df[datetime_col] = pd.to_datetime(df[datetime_col])
#         df = df.sort_values(by=datetime_col)
#         df = ForecastPipeline.create_time_features(df, datetime_col, target_col)
#         features = [col for col in selected_features if col not in (group_cols or []) + [datetime_col]]
#         df_processed = self.preprocess(df, features, target_col)
#         self.features = [col for col in df_processed.columns if col != target_col]
#         X = df_processed[self.features]
#         y = df_processed[target_col]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#         self.model = xgb.XGBRegressor(**self.xgb_params)
#         self.model.fit(X_train, y_train)
#         y_pred = self.model.predict(X_test)
#         mape = mean_absolute_percentage_error(y_test, y_pred)
#         print("✅ Model training completed. MAPE:", mape)
#         print("✅ Trained Features:", self.features)
#         return self.model, mape

#     def forecast_for_combination(self, history_df, future_df, target_col, datetime_col, selected_features):
#         preds = []
#         last_known = history_df.tail(max(5, max(1, self.forecast_value))).copy()
#         numerical_features_to_carry = [
#             f for f in selected_features if f not in [datetime_col, target_col] and f not in self.encoders
#         ]

#         for i in range(self.forecast_value):
#             current_future_date = future_df.iloc[i][datetime_col]
#             temp_future_row = {'datetime': current_future_date}

#             # Carry forward categorical features
#             for col in [f for f in selected_features if f in self.encoders]:
#                 temp_future_row[col] = future_df.iloc[i][col]

#             # Carry forward numerical features
#             for col in numerical_features_to_carry:
#                 temp_future_row[col] = future_df.iloc[i][col]

#             # Add time features
#             time_features = self.add_time_features(pd.DataFrame([temp_future_row]), 'datetime').iloc[0]
#             for tf_col, tf_val in time_features.items():
#                 if tf_col not in temp_future_row:
#                     temp_future_row[tf_col] = tf_val

#             # Lags - use the history of this specific combination
#             lag_1 = last_known[target_col].iloc[-1] if not last_known.empty else np.nan
#             lag_2 = last_known[target_col].iloc[-2] if len(last_known) >= 2 else np.nan
#             temp_future_row["lag_1"] = lag_1
#             temp_future_row["lag_2"] = lag_2
#             temp_future_row["rolling_mean_3"] = last_known[target_col].rolling(3).mean().iloc[-1] if len(last_known) >= 3 else np.nan
#             temp_future_row["rolling_mean_5"] = last_known[target_col].rolling(5).mean().iloc[-1] if len(last_known) >= 5 else np.nan

#             # Prepare input DataFrame for the current forecast step
#             forecast_input_df = pd.DataFrame([temp_future_row])
#             features_for_prediction = [f for f in selected_features if f != datetime_col and f != target_col]

#             # --- Preprocessing within the forecasting loop ---
#             processed_forecast_input = forecast_input_df[features_for_prediction].copy()

#             # Label Encode
#             for col in [f for f in features_for_prediction if f in self.encoders]:
#                 le = self.encoders[col]
#                 processed_forecast_input[col] = processed_forecast_input[col].astype(str).map(dict(zip(le.classes_, le.transform(le.classes_)))).fillna(-1).astype(int)

#             # One-Hot Encode
#             processed_forecast_input = pd.get_dummies(processed_forecast_input,
#                                                       columns=[f for f in features_for_prediction if f in self.encoders],
#                                                       prefix=[f for f in features_for_prediction if f in self.encoders],
#                                                       dummy_na=False)

#             # Create the final input DataFrame with all training columns
#             input_data = pd.DataFrame(columns=self.final_training_columns)

#             # Populate the input data with the processed features
#             for col in processed_forecast_input.columns:
#                 if col in input_data.columns:
#                     input_data[col] = processed_forecast_input[col]

#             # Fill any missing columns with 0 (for one-hot encoded) or NaN (for numerical, will be handled by scaler)
#             input_data = input_data.fillna(0) # For one-hot encoded columns

#             print("✅ Final Input Data Columns Before Scaling:", input_data.columns.tolist())

#             # Scale the input data
#             numerical_cols_to_scale = [col for col in self.numerical_cols_scaled if col in input_data.columns]
#             if numerical_cols_to_scale:
#                 input_data[numerical_cols_to_scale] = self.scaler.transform(input_data[numerical_cols_to_scale])

#             # Predict the target
#             if self.model and not input_data.empty:
#                 scaled_input = input_data[self.final_training_columns].values  # Ensure correct column order
#                 y_pred = self.model.predict(scaled_input)[0]
#                 preds.append(y_pred)

#                 # Update last_known
#                 next_date = current_future_date
#                 next_features = {f: temp_future_row[f] for f in selected_features if f != target_col and f != datetime_col}
#                 new_row = {datetime_col: next_date, target_col: y_pred, **next_features}
#                 last_known = pd.concat([last_known, pd.DataFrame([new_row])], ignore_index=True)
#             else:
#                 preds.append(np.nan)

#         return np.array(preds)

#     def run(self, df, selected_features, target_col):
#         # detect datetime
#         datetime_col, score = self.detect_datetime_column(df, df.columns.tolist())
#         if datetime_col is None:
#             raise ValueError("No valid datetime column found.")
#         df[datetime_col] = pd.to_datetime(df[datetime_col], format='mixed')
#         df = df.sort_values(by=datetime_col)

#         # train
#         model, mape = self.train(df, datetime_col, target_col, selected_features)

#         categorical_features = [f for f in selected_features if f in self.encoders]
#         grouping_features = categorical_features

#         unique_combinations_df = df[grouping_features].drop_duplicates().reset_index(drop=True)
#         print(f"Number of unique combinations: {len(unique_combinations_df)}") 
#         all_forecasts = []

#         for index, combo_row in unique_combinations_df.iterrows():
#             combo = combo_row.to_dict()
#             # Filter the original data for the current combination
#             condition = True
#             for col, val in combo.items():
#                 condition = condition & (df[col] == val)
#             combo_df = df[condition].copy()

#             if combo_df.empty:
#                 continue  # Skip if no data for this combination

#             last_known = combo_df.tail(1).copy()
#             if last_known.empty:
#                 continue

#             last_timestamp = pd.Timestamp(last_known[datetime_col].values[0])
#             freq_str = self._freq_str()
#             future_dates = pd.date_range(start=last_timestamp, periods=self.forecast_value, freq=freq_str)

#             future_combo_df = pd.DataFrame({datetime_col: future_dates})
#             for col, val in combo.items():
#                 future_combo_df[col] = val

#             # Determine numerical features to carry forward for this combination
#             numerical_features_to_carry_combo = [
#                 f for f in selected_features if f not in [datetime_col, target_col] and f not in self.encoders
#             ]

#             # Carry forward the last known values of numerical features
#             if not combo_df.empty:
#                 last_numerical_values = combo_df[numerical_features_to_carry_combo].iloc[-1].to_dict()
#                 for col in numerical_features_to_carry_combo:
#                     if col not in future_combo_df.columns:
#                         future_combo_df[col] = last_numerical_values.get(col)
#                     else:
#                         future_combo_df[col] = last_numerical_values.get(col) # Handle if it exists in combo
#             print(f"Forecasting for combination: {combo}")
#             # Perform forecasting for this specific combination
#             combo_forecast = self.forecast_for_combination(combo_df, future_combo_df, target_col, datetime_col, selected_features)
#             combo_forecast_df = pd.DataFrame({datetime_col: future_dates, target_col: combo_forecast})
#             for col, val in combo.items():
#                 combo_forecast_df[col] = val
#             all_forecasts.append(combo_forecast_df)

#         if all_forecasts:
#             results_df = pd.concat(all_forecasts, ignore_index=True)
#         else:
#             results_df = pd.DataFrame()

#         return results_df, mape

#     def _freq_str(self):
#         freq_map = {
#             '5 minutes': '5T',
#             'hourly':    'H',
#             'daily':     'D',
#             'monthly':   'M',
#             'yearly':    'Y'
#         }
#         key = str(self.period_option).strip().lower()
#         return freq_map.get(key, 'D')







# class ForecastPipeline:
#     def __init__(
#         self,
#         xgb_params=None,
#         fine_tune_method="None",
#         period_option="Daily",
#         forecast_value=10
#     ):
#         # Default XGBoost params
#         if xgb_params is None:
#             xgb_params = {
#                 "base_score": 0.5,
#                 "booster": "gbtree",
#                 "objective": "reg:squarederror",
#                 "n_estimators": 500,
#                 "learning_rate": 0.01,
#                 "max_depth": 5,
#                 "min_child_weight": 5,
#                 "reg_lambda": 10,
#                 "reg_alpha": 5
#             }
#         self.xgb_params = xgb_params
#         self.fine_tune_method = fine_tune_method
#         self.period_option = period_option
#         self.forecast_value = forecast_value

#     @staticmethod
#     def robust_parse_date(val):
#         s = str(val).strip()
#         if not s:
#             return pd.NaT
#         if s.isdigit() and len(s) == 8:
#             try:
#                 dt = datetime.strptime(s, "%Y%m%d")
#                 if 1900 <= dt.year <= 2100:
#                     return pd.Timestamp(dt)
#             except ValueError:
#                 return pd.NaT
#         if re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", s):
#             return pd.NaT
#         if not any(d in s for d in ["-", "/", "."]):
#             return pd.NaT
#         try:
#             dt = parse(s, fuzzy=False)
#             if 1900 <= dt.year <= 2100:
#                 return pd.Timestamp(dt)
#         except Exception:
#             return pd.NaT
#         return pd.NaT

#     @staticmethod
#     def detect_datetime_column(df, selected_cols, threshold=0.8, exclude_non_date_text=True):
#         for col in selected_cols:
#             if pd.api.types.is_datetime64_any_dtype(df[col]):
#                 return col, 1.0
#         best_col, best_ratio = None, 0.0
#         for col in selected_cols:
#             if exclude_non_date_text:
#                 mixed = df[col].astype(str).str.contains(r"[A-Za-z].*\d|\d.*[A-Za-z]").mean()
#                 if mixed > 0.5:
#                     continue
#             parsed = df[col].apply(ForecastPipeline.robust_parse_date)
#             ratio = parsed.notna().mean()
#             if ratio > best_ratio:
#                 best_ratio, best_col = ratio, col
#         return (best_col, best_ratio) if best_ratio >= threshold else (None, best_ratio)

#     @staticmethod
#     def add_time_features(df, datetime_col):
#         dt = df[datetime_col]
#         df['year'] = dt.dt.year
#         df['month'] = dt.dt.month
#         df['day'] = dt.dt.day
#         df['weekday'] = dt.dt.weekday
#         df['hour'] = dt.dt.hour
#         df['minute'] = dt.dt.minute
#         df['second'] = dt.dt.second
#         df['week_number'] = dt.dt.isocalendar().week.astype(int)
#         df['is_month_start'] = dt.dt.is_month_start.astype(int)
#         df['is_month_end'] = dt.dt.is_month_end.astype(int)
#         df['quarter'] = dt.dt.quarter
#         return df

#     @staticmethod
#     def create_time_features(df, datetime_col, target_col, default_lags=[1, 2], default_rolling=[3, 5]):
#         df = df.copy()
#         df = ForecastPipeline.add_time_features(df, datetime_col)
#         for lag in default_lags:
#             df[f'lag_{lag}'] = df[target_col].shift(lag)
#         for window in default_rolling:
#             df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
#         return df

#     def run(self, df, feature_cols, target_col):
#         # Early exit
#         if self.forecast_value == 0:
#             return pd.DataFrame(columns=feature_cols + ['forecast']), None

#         # 1. Detect datetime
#         datetime_col, ratio = self.detect_datetime_column(df, feature_cols)
#         if datetime_col is None:
#             raise ValueError("Could not detect a datetime column among the selected features.")
#         df = df.copy()
#         df[datetime_col] = df[datetime_col].apply(self.robust_parse_date)
#         global_max = df[datetime_col].max()

#         # 2. Define grouping & additional features
#         group_features = [c for c in feature_cols if c != datetime_col and not pd.api.types.is_numeric_dtype(df[c])]
        
#         # print("===================GROUP FEATURE :====;",group_features)
#         additional_features = [c for c in feature_cols if c != datetime_col and c not in group_features]
       
#         # print("======================++++++ ADD FEATURES +++++++++++++++==;", additional_features)

#         # 3. Build group IDs
#         if group_features:
#             join_sym = '__Jos__'
#             df['__group_id__'] = df[group_features].astype(str).agg(join_sym.join, axis=1)
#             group_counts = df['__group_id__'].value_counts()
#             too_small = group_counts[group_counts < 5]
            
#             if len(too_small) / len(group_counts) > 0.5:
#                 print("\n[WARNING] Too many fragmented groups. Evaluating overly specific grouping columns...\n")
                
#                 # Try dropping the most specific group feature one by one
#                 dropped_cols = []
#                 reduced_groups = group_features.copy()
                
#                 while reduced_groups:
#                     test_group_id = df[reduced_groups].astype(str).agg(join_sym.join, axis=1)
#                     test_counts = test_group_id.value_counts()
#                     test_too_small = test_counts[test_counts < 5]

#                     if len(test_too_small) / len(test_counts) <= 0.5:
#                         print(f"[INFO] Dropped group column(s): {dropped_cols}")
#                         df['__group_id__'] = test_group_id
#                         group_features = reduced_groups
#                         break
#                     else:
#                         col_to_drop = reduced_groups[-1]
#                         dropped_cols.append(col_to_drop)
#                         reduced_groups = reduced_groups[:-1]
                
#                 # If all are dropped, fallback to no grouping
#                 if not reduced_groups:
#                     print("[INFO] All group columns dropped due to over-fragmentation.")
#                     df['__group_id__'] = 'ALL'
#                     group_features = []



#             groups = df['__group_id__'].unique()
#         else:
#             print(f"[INFO] Proceeding with {len(groups)} group(s).")

#         all_forecasts = []
#         mape_list = []

#         # freq map
#         freq_map = {'hourly':'H','daily':'D','5 minutes':'5T','weekly':'W','monthly':'M','yearly':'Y'}
#         freq = freq_map.get(self.period_option.lower(), 'D')

#         for grp in groups:
#             df_grp = df[df['__group_id__'] == grp].copy() if grp is not None else df.copy()
#             df_grp = df_grp.dropna(subset=[datetime_col])
#             df_grp.sort_values(by=datetime_col, inplace=True)


#             if len(df_grp) < 5:
#                 print(f"[SKIPPED] Group {grp}: only {len(df_grp)} rows.")
#                 continue

            
#             new_group= group_features+additional_features
#             # print("==================++++++",new_group)
#             # Prepare features
#             Xf = df_grp[new_group].copy()
#             cats = Xf.select_dtypes(include=['object']).columns.tolist()
#             # print("=================CATS+++++++++++;",cats)
#             nums = [c for c in Xf if c not in cats]
#             for c in cats:
#                 Xf[c] = LabelEncoder().fit_transform(Xf[c])
#             Xn = pd.DataFrame(StandardScaler().fit_transform(Xf[nums]), columns=nums, index=Xf.index) if nums else pd.DataFrame(index=Xf.index)
#             # print("#$#$#$#$ NUMb====",Xn)
#             Xc = Xf[cats].copy() if cats else pd.DataFrame(index=Xf.index)
#             Xp = pd.concat([Xc, Xn], axis=1)

#             # print("--------------------- XP========",Xp)

#             proc = pd.concat([df_grp[[datetime_col]], Xp, df_grp[[target_col]]], axis=1)
#             # print("--------------------- proc========$$$$$$",proc)
#             proc = self.create_time_features(proc, datetime_col, target_col)
#             # print("--------------------- proc========#####################$$$$$$",proc)
#             proc.dropna(inplace=True)
#             if len(proc) < 5:
#                 print(f"Group {grp} dropped due to NaNs in time features. Remaining rows: {len(proc)}")
#                 continue

#             # if len(proc) < 10:
#             #     continue

#             X_full = proc.drop(columns=[datetime_col, target_col])
#             # print("========++====++++===+++++:X_full",X_full)
#             y_full = proc[target_col]
#             # print("================y_full========++++",y_full)
#             X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
#             model = xgb.XGBRegressor(**self.xgb_params)

#             # tuning
#             search = None
#             pg = {"max_depth":[3,5,7],"learning_rate":[0.01,0.05,0.1],"n_estimators":[100,300,500],"min_child_weight":[1,3,5],"reg_lambda":[0,10,20],"reg_alpha":[0,5,10]}
#             if self.fine_tune_method=="Grid Search": search = GridSearchCV(model, pg, cv=3, n_jobs=-1, verbose=1)
#             elif self.fine_tune_method=="Random Search": search = RandomizedSearchCV(model, pg, cv=3, n_iter=10, n_jobs=-1, verbose=1, random_state=42)
#             elif self.fine_tune_method=="Bayesian Optimization" and BayesSearchCV: search = BayesSearchCV(model, pg, cv=3, n_iter=15, n_jobs=-1, verbose=1, random_state=42)
#             if search: search.fit(X_tr, y_tr); model=search.best_estimator_
#             else: model.fit(X_tr, y_tr, eval_set=[(X_tr,y_tr),(X_te,y_te)], verbose=False)

#             mape_list.append(mean_absolute_percentage_error(y_tr, model.predict(X_tr)))

#             # exogenous
#             exog_feats = [c for c in X_full if c not in ['lag_1','lag_2']]
#             # print("************************* Exog_Feats ******************",exog_feats)
#             exog_fc = {}
            
#             for feat in exog_feats:
#                 s = proc[feat].dropna()
#                 # print("------------------ SSSSS--------------",s)
#                 if len(s)<2: exog_fc[feat]=np.repeat(s.iloc[-1], self.forecast_value)
#                 else:
#                     try: exog_fc[feat]=ARIMA(s,order=(1,1,1)).fit(disp=False).forecast(self.forecast_value)
#                     except: exog_fc[feat]=np.repeat(s.iloc[-1], self.forecast_value)
#             # print("**^*^^*^*^*^*^*^* exog_fc ================",exog_fc)
#             last = proc.iloc[-1].drop([datetime_col, target_col]).copy()
#             fo = []
#             for i, dt in enumerate(pd.date_range(start=global_max, periods=self.forecast_value+1, freq=freq)[1:]):
#                 tf = self.add_time_features(pd.DataFrame({datetime_col:[dt]}), datetime_col).iloc[0].drop(datetime_col)
#                 last.update(tf)
#                 x = last[X_full.columns].values.reshape(1,-1)
#                 # print("========================= X_full ##############",x)
#                 p = model.predict(x)[0]
#                 # print("==========================PREDICTED===============",p)
#                 # shift
#                 if 'lag_1' in last and 'lag_2' in last: last['lag_2']=last['lag_1']; last['lag_1']=p
#                 for feat in exog_feats: last[feat]=exog_fc[feat][i]
#                 rec = {datetime_col:dt, 'forecast':p}
#                 # print("=====++++++++++++ REC ++++++++++++++",rec)
#                 # add group and additional cols
#                 if group_features:
#                     for gf in group_features: rec[gf]=df_grp[gf].iloc[0]
#                 for af in additional_features: rec[af]=df_grp[af].iloc[-1]
#                 fo.append(rec)

#             if not fo:
#                 print(f"[WARNING] No forecasts produced for group {grp}. Check data consistency.")   

#             all_forecasts.extend(fo)

#         if not all_forecasts:
#             print("=== Forecasting Summary ===")
#             print(f"Total groups found: {len(groups)}")
#             print(f"Successful forecasts generated: {len(all_forecasts)}")
#             raise ValueError("No forecast results generated for any group.")
#         final = pd.DataFrame(all_forecasts)
#         return final, (sum(mape_list)/len(mape_list) if mape_list else None)























# class ForecastPipeline:
#     def __init__(
#         self,
#         xgb_params=None,
#         fine_tune_method="None",
#         period_option="Daily",
#         forecast_value=10
#     ):
#         if xgb_params is None:
#             xgb_params = {
#                 "base_score": 0.5,
#                 "booster": "gbtree",
#                 "objective": "reg:squarederror",
#                 "n_estimators": 500,
#                 "learning_rate": 0.01,
#                 "max_depth": 5,
#                 "min_child_weight": 5,
#                 "reg_lambda": 10,
#                 "reg_alpha": 5
#             }
#         self.xgb_params = xgb_params
#         self.fine_tune_method = fine_tune_method
#         self.period_option = period_option
#         self.forecast_value = forecast_value
#         self.model = None
#         self.scaler = StandardScaler()
#         self.encoders = {}

#     @staticmethod
#     def robust_parse_date(val):
#         s = str(val).strip()
#         if not s:
#             return pd.NaT
#         if s.isdigit() and len(s) == 8:
#             try:
#                 dt = datetime.strptime(s, "%Y%m%d")
#                 if 1900 <= dt.year <= 2100:
#                     return pd.Timestamp(dt)
#             except ValueError:
#                 return pd.NaT
#         if re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", s):
#             return pd.NaT
#         if not any(d in s for d in ["-", "/", "."]):
#             return pd.NaT
#         try:
#             dt = parse(s, fuzzy=False)
#             if 1900 <= dt.year <= 2100:
#                 return pd.Timestamp(dt)
#         except Exception:
#             return pd.NaT
#         return pd.NaT

#     @staticmethod
#     def detect_datetime_column(df, selected_cols, threshold=0.8, exclude_non_date_text=True):
#         for col in selected_cols:
#             if pd.api.types.is_datetime64_any_dtype(df[col]):
#                 return col, 1.0
#         best_col, best_ratio = None, 0.0
#         for col in selected_cols:
#             if exclude_non_date_text:
#                 mixed = df[col].astype(str).str.contains(r"[A-Za-z].*\d|\d.*[A-Za-z]").mean()
#                 if mixed > 0.5:
#                     continue
#             parsed = df[col].apply(ForecastPipeline.robust_parse_date)
#             ratio = parsed.notna().mean()
#             if ratio > best_ratio:
#                 best_ratio, best_col = ratio, col
#         return (best_col, best_ratio) if best_ratio >= threshold else (None, best_ratio)

#     @staticmethod
#     def add_time_features(df, datetime_col):
#         dt = df[datetime_col]
#         df['year'] = dt.dt.year
#         df['month'] = dt.dt.month
#         df['day'] = dt.dt.day
#         df['weekday'] = dt.dt.weekday
#         df['hour'] = dt.dt.hour
#         df['minute'] = dt.dt.minute
#         df['second'] = dt.dt.second
#         df['week_number'] = dt.dt.isocalendar().week.astype(int)
#         df['is_month_start'] = dt.dt.is_month_start.astype(int)
#         df['is_month_end'] = dt.dt.is_month_end.astype(int)
#         df['quarter'] = dt.dt.quarter
#         return df

#     @staticmethod
#     def create_time_features(df, datetime_col, target_col, default_lags=[1, 2], default_rolling=[3, 5]):
#         df = df.copy()
#         df = ForecastPipeline.add_time_features(df, datetime_col)
#         for lag in default_lags:
#             df[f'lag_{lag}'] = df[target_col].shift(lag)
#         for window in default_rolling:
#             df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
#         return df
    

#     def preprocess(self, df, features, target_col):
#         df = df.dropna(subset=[target_col])
#         for col in features:
#             if df[col].dtype == 'object' or df[col].dtype.name == 'category':
#                 le = LabelEncoder()
#                 df[col] = le.fit_transform(df[col])
#                 self.encoders[col] = le
#         df[features] = self.scaler.fit_transform(df[features])
#         return df

#     def train(self, df, datetime_col, target_col, selected_features, group_cols=None):
#         df[datetime_col] = pd.to_datetime(df[datetime_col])
#         df = df.sort_values(by=datetime_col)
#         df = ForecastPipeline.create_time_features(df, datetime_col, target_col)
#         features = [col for col in selected_features if col not in (group_cols or []) + [datetime_col]]
#         df = self.preprocess(df, features, target_col)
#         X = df[features]
#         y = df[target_col]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#         self.model = xgb.XGBRegressor(**self.xgb_params)
#         self.model.fit(X_train, y_train)
#         self.features = features
#         y_pred = self.model.predict(X_test)
#         mape = mean_absolute_percentage_error(y_test, y_pred)
#         print("✅ Model training completed. MAPE:", mape)
#         return self.model, mape

#     def forecast(self, df, future_df, target_col):
#         for col, encoder in self.encoders.items():
#             if col in future_df.columns:
#                 future_df[col] = future_df[col].astype(str)
#                 df[col] = df[col].astype(str)
#                 all_vals = set(df[col]).union(set(future_df[col]))
#                 new_classes = np.unique(list(encoder.classes_) + list(all_vals))
#                 if set(new_classes) != set(encoder.classes_):
#                     encoder = LabelEncoder()
#                     encoder.fit(new_classes)
#                     self.encoders[col] = encoder
#                     df[col] = encoder.transform(df[col])
#                     self.model.fit(df[self.features], df[target_col])
#                 future_df[col] = encoder.transform(future_df[col])

#         last_known = df.tail(max(5, max(1, self.forecast_value)))
#         future_df["lag_1"] = last_known[target_col].iloc[-1]
#         future_df["lag_2"] = last_known[target_col].iloc[-2] if len(last_known)>=2 else np.nan
#         future_df["rolling_mean_3"] = last_known[target_col].rolling(3).mean().iloc[-1]
#         future_df["rolling_mean_5"] = last_known[target_col].rolling(5).mean().iloc[-1]

#         future_df[self.features] = self.scaler.transform(future_df[self.features])
#         preds = self.model.predict(future_df[self.features])
#         return preds

#     def run(self, df, selected_features, target_col):
#         datetime_col, score = self.detect_datetime_column(df, df.columns.tolist())
#         if datetime_col is None:
#             raise ValueError("No valid datetime column found.")
#         df[datetime_col] = pd.to_datetime(df[datetime_col])
#         df = df.sort_values(by=datetime_col)

#         model, mape = self.train(df, datetime_col, target_col, selected_features)

#         self.encoders = {}
#         for col in selected_features:
#             if df[col].dtype == 'object':
#                 le = LabelEncoder()
#                 df[col] = le.fit_transform(df[col])
#                 self.encoders[col] = le
#         self.scaler = StandardScaler()
#         df[self.features] = self.scaler.fit_transform(df[self.features])

#         last_known = df.tail(1).copy()
#         last_timestamp = pd.Timestamp(last_known[datetime_col].values[0])

#         freq_str = self._freq_str()
#         offset   = pd.tseries.frequencies.to_offset(freq_str)

#         future_dates = pd.date_range(
#             start   = last_timestamp + offset,
#             periods = self.forecast_value,
#             freq    = freq_str
#         )

#         future_df = pd.DataFrame({ datetime_col: future_dates })
#         for feat, val in zip(selected_features, last_known[selected_features].values[0]):
#             future_df[feat] = val

#         future_df = self.add_time_features(future_df, datetime_col)
#         preds = self.forecast(df, future_df, target_col)

#         future_df[target_col] = preds
#         # drop any duplicate columns to avoid Streamlit errors
#         future_df = future_df.loc[:, ~future_df.columns.duplicated()]

#         # order output columns: datetime, target, then the rest of features
#         output_cols = [datetime_col, target_col] + [c for c in selected_features if c not in (datetime_col, target_col)]
#         return future_df[output_cols], mape

#     def _freq_str(self):
#         freq_map = {
#             "5 Minutes": "5T",
#             "Hourly":    "H",
#             "Daily":     "D",
#             "Monthly":   "M",
#             "Yearly":    "Y",
#         }
#         return freq_map.get(self.period_option, "D")
    















    # def preprocess(self, df, features, target_col):
    #     df = df.dropna(subset=[target_col])
    #     print("=============================DF==========",df.head())
    #     print("=============================DF==========",df.shape)
    #     # print("=============================DF==========",df.cloumns)

    #     print("################# features ====$$$$$$$$$$$$",features)
    #     print("################# target_col ====$$$$$$$$$$$$",target_col)
    #     for col in features:
    #         if df[col].dtype == 'object' or df[col].dtype.name == 'category':
    #             le = LabelEncoder()
    #             df[col] = le.fit_transform(df[col])
    #             self.encoders[col] = le

    #     df[features] = self.scaler.fit_transform(df[features])
    #     return df

    # def train(self, df, datetime_col, target_col, selected_features, group_cols=None):
    #     df[datetime_col] = pd.to_datetime(df[datetime_col])
    #     df = df.sort_values(by=datetime_col)
    #     df = ForecastPipeline.create_time_features(df, datetime_col, target_col)

    #     # Only use selected features (excluding group cols)
    #     # features = [col for col in selected_features if col not in (group_cols or [])]
    #     features = [col for col in selected_features if col not in (group_cols or []) + [datetime_col]]

    #     df = self.preprocess(df, features, target_col)

    #     X = df[features]
    #     y = df[target_col]
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    #     self.model = xgb.XGBRegressor(**self.xgb_params)
    #     self.model.fit(X_train, y_train)
    #     self.features = features  # Save only selected features

    #     y_pred = self.model.predict(X_test)
    #     mape = mean_absolute_percentage_error(y_test, y_pred)
    #     print("✅ Model training completed. MAPE:", mape)
    #     return self.model, mape


    # def forecast(self, df, future_df, target_col):
    #     # Fill categorical values using encoders
    #     for col, encoder in self.encoders.items():
    #         if col in future_df.columns:
    #             # Convert both to string before any checks
    #             future_df[col] = future_df[col].astype(str)
    #             df[col] = df[col].astype(str)

    #             df_values = set(df[col].unique())
    #             future_values = set(future_df[col].unique())

    #             known_classes = set(encoder.classes_)
    #             all_new_values = df_values.union(future_values)
    #             unseen_values = all_new_values - known_classes

    #             if unseen_values:
    #                 # print(f"⚠️ Updating LabelEncoder for column '{col}' with unseen values: {unseen_values}")

    #                 # Combine all classes: old + new
    #                 existing = [str(cls) for cls in encoder.classes_]
    #                 new_vals = [str(val) for val in unseen_values]
    #                 all_classes = np.unique(existing + new_vals + list(df_values))  # Include df values

    #                 # Re-fit encoder
    #                 new_encoder = LabelEncoder()
    #                 new_encoder.fit(all_classes)
    #                 self.encoders[col] = new_encoder

    #                 # Encode both
    #                 df[col] = new_encoder.transform(df[col])
    #                 future_df[col] = future_df[col].apply(lambda x: new_encoder.transform([x])[0])

    #                 # Retrain model
    #                 # print(f"🔁 Retraining model after updating encoder for '{col}'...")
    #                 self.model.fit(df[self.features], df[target_col])
    #             else:
    #                 # Use original encoder
    #                 future_df[col] = future_df[col].apply(lambda x: encoder.transform([x])[0])




    #     # Add engineered features based on last known values
    #     last_known = df.tail(5)  # get enough rows for lag and rolling
    #     future_df["lag_1"] = last_known[target_col].shift(1).iloc[-1]
    #     future_df["lag_2"] = last_known[target_col].shift(2).iloc[-1]
    #     future_df["rolling_mean_3"] = last_known[target_col].rolling(3).mean().iloc[-1]
    #     future_df["rolling_mean_5"] = last_known[target_col].rolling(5).mean().iloc[-1]

    #     print("=-=-=-=-=-=-=-=-=-=-=-=-= future_df===:", future_df)

    #     # Validate all required features exist
    #     missing_cols = [col for col in self.features if col not in future_df.columns]
    #     if missing_cols:
    #         raise ValueError(f"Missing columns in future_df: {missing_cols}")

    #     future_df[self.features] = self.scaler.transform(future_df[self.features])
    #     predictions = self.model.predict(future_df[self.features])
    #     return predictions


    # def run(self, df, selected_features, target_col):
    #     datetime_col, score = self.detect_datetime_column(df, df.columns.tolist())
    #     if datetime_col is None:
    #         raise ValueError("No valid datetime column found.")

    #     df[datetime_col] = pd.to_datetime(df[datetime_col])
    #     df = df.sort_values(by=datetime_col)

    #     # Fit model using only selected features
    #     model, mape = self.train(df, datetime_col, target_col, selected_features)

    #     # Store encoders & scaler during training
    #     self.encoders = {}
    #     for col in selected_features:
    #         if df[col].dtype == 'object':
    #             encoder = LabelEncoder()
    #             df[col] = encoder.fit_transform(df[col])
    #             self.encoders[col] = encoder

    #     self.scaler = StandardScaler()
    #     df[self.features] = self.scaler.fit_transform(df[self.features])

    #     # Create future_df from last known row
    #     last_known = df.tail(1).copy()
    #     print("===========================================$$$ Last_known:",last_known)
    #     # last_timestamp = last_known[datetime_col].values[0]
    #     last_timestamp = pd.Timestamp(last_known[datetime_col].values[0])
    #     print("----------------------------last_timestamp--------------",last_timestamp)
    #     print("➡️ Forecasting frequency string:+++++++++++++++", self._freq_str())
    #     future_dates = pd.date_range(start=last_timestamp, periods=self.forecast_value + 1, freq=self._freq_str())
    #     print("=====================----- futures_date --------------",future_dates)

    #     future_df = pd.DataFrame({datetime_col: future_dates})
    #     print("======================^^^^^^^^^^^^future_df",future_df)
    #     base_values = last_known[selected_features].values[0]
    #     # base_values = pd.Timestamp(last_known[selected_features].values[0])
    #     print("][[][][][][][]][][[]====]",base_values)
    #     for feat, val in zip(selected_features, base_values):
    #         future_df[feat] = val
    #     print("-=-=-=-=-=-=-==-=-=-=-=-= future_df[feat]=-=-=-",future_df[feat])

    #     future_df = self.add_time_features(future_df, datetime_col)
    #     print("==================****** future_df:-==",future_df)
    #     forecast_values = self.forecast(df, future_df, target_col)
    #     print("=========== forecasted values ====-------;",forecast_values)

    #     future_df[target_col] = forecast_values
    #     future_df = future_df[[datetime_col, target_col] + selected_features]
    #     future_df = future_df.loc[:, ~future_df.columns.duplicated()]
    #     return future_df, mape


    # def _freq_str(self):
    #     freq_map = {
    #         "5 Minutes": "5T",
    #         "Hourly": "H",
    #         "Daily": "D",
    #         "Monthly": "M",
    #         "Yearly": "Y"
    #     }
    #     return freq_map.get(self.period_option.lower(),"D")


# "5 Minutes", "Hourly", "Daily", "Monthly", "Yearly"













    # def run(self, df, feature_cols, target_col):
    #     # 1. Detect datetime column.
    #     datetime_col, best_ratio = self.detect_datetime_column(df, feature_cols)
    #     if datetime_col is None:
    #         raise ValueError("Could not detect a datetime column among the selected features.")

    #     # 2. Remove additional datetime-like columns.
    #     datetime_threshold = 0.8
    #     additional_features = [col for col in feature_cols if col != datetime_col and 
    #                            df[col].apply(self.robust_parse_date).notnull().mean() < datetime_threshold]
        
    #     print("==============================additional_features==================",additional_features)

    #     # 3. Convert datetime column.
    #     df[datetime_col] = df[datetime_col].apply(self.robust_parse_date)
    #     if df[datetime_col].isnull().any():
    #         raise ValueError("Datetime conversion error: Some dates could not be parsed.")
    #     cols_order = [datetime_col] + [col for col in df.columns if col != datetime_col]
    #     df = df[cols_order]

    #     # 4. Process additional features.
    #     X_features = df[additional_features].copy() if additional_features else pd.DataFrame()
    #     print("==================++++++++++X_features=======",X_features)
    #     cat_cols = X_features.select_dtypes(include=['object']).columns.tolist()
    #     print("+++++++++++++++++++++++++++++++cat_cols=========+++++",cat_cols)
    #     num_cols = [col for col in X_features.columns if col not in cat_cols]
    #     print("---------------------------------num_cols-----------------",num_cols)
    #     encoders = {}
    #     for col in cat_cols:
    #         le = LabelEncoder()
    #         X_features[col] = le.fit_transform(X_features[col])
    #         encoders[col] = le
    #     if num_cols:
    #         scaler = StandardScaler()
    #         X_features_scaled_num = pd.DataFrame(scaler.fit_transform(X_features[num_cols]), columns=num_cols)
    #     else:
    #         X_features_scaled_num = pd.DataFrame()
    #     X_features_scaled_cat = X_features[cat_cols].copy() if cat_cols else pd.DataFrame()
    #     if not X_features_scaled_cat.empty and not X_features_scaled_num.empty:
    #         X_features_processed = pd.concat([X_features_scaled_cat, X_features_scaled_num], axis=1)
    #     elif not X_features_scaled_cat.empty:
    #         X_features_processed = X_features_scaled_cat.copy()
    #     else:
    #         X_features_processed = X_features_scaled_num.copy()

    #     # 5. Prepare target.
    #     y = df[target_col].copy()

    #     # 6. Combine datetime, processed features, and target.
    #     data = pd.concat([df[[datetime_col]], X_features_processed, y], axis=1)
    #     print("--==-==-=-=-=-=-==--=-=-=data=====",data)
    #     # 7. Create time features.
    #     data = self.create_time_features(data, datetime_col, target_col)
    #     print("--==-==-=-=-=-=-==--=-=-***************8=data=====",data)
    #     print("--==-==-=-=-=-=-==--=-=-***************8=data=====",data.shape)
    #     data.dropna(inplace=True)

    #     X_full = data.drop(columns=[datetime_col, target_col])
    #     print("+*+*+*+*+*+*+*+*+**++*+*+*+*+* X_Full++++++",X_full)
    #     print("+*+*+*+*+*+*+*+*+**++*+*+*+*+* X_Full++++++",X_full.columns)
    #     print("+*+*+*+*+*+*+*+*+**++*+*+*+*+* X_Full++++++",X_full.shape)
    #     y_full = data[target_col]
    #     print("+*+*+*+*+*+*+*+*+**++*+*+*+*+* y_full++++++",y_full)
    #     X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    #     # 8. Train the XGBoost model.
    #     model = xgb.XGBRegressor(**self.xgb_params)
    #     if self.fine_tune_method != "None":
    #         param_grid = {
    #             "max_depth": [3, 5, 7],
    #             "learning_rate": [0.01, 0.05, 0.1],
    #             "n_estimators": [100, 300, 500],
    #             "min_child_weight": [1, 3, 5],
    #             "reg_lambda": [0, 10, 20],
    #             "reg_alpha": [0, 5, 10]
    #         }
    #         if self.fine_tune_method == "Grid Search":
    #             model_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    #         elif self.fine_tune_method == "Random Search":
    #             model_search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=10, n_jobs=-1, verbose=1, random_state=42)
    #         elif self.fine_tune_method == "Bayesian Optimization":
    #             if BayesSearchCV is None:
    #                 raise ImportError("Bayesian Optimization requires scikit-optimize. Please install it.")
    #             model_search = BayesSearchCV(model, param_grid, cv=3, n_iter=15, n_jobs=-1, verbose=1, random_state=42)
    #         else:
    #             model_search = None

    #         if model_search is not None:
    #             model_search.fit(X_train, y_train)
    #             model = model_search.best_estimator_
    #         else:
    #             model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
    #     else:
    #         model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

    #     # 9. Forecast exogenous features using ARIMA.
    #     exog_feats = [col for col in X_full.columns if col not in ['lag_1','lag_2']]
    #     exog_forecasts = {}
    #     for feat in exog_feats:
    #         try:
    #             series = data[feat].dropna()
    #             arima_model = ARIMA(series, order=(1,1,1)).fit()
    #             forecast_series = arima_model.forecast(steps=self.forecast_value)
    #             exog_forecasts[feat] = forecast_series.values
    #         except Exception:
    #             exog_forecasts[feat] = np.repeat(data[feat].iloc[-1], self.forecast_value)

    #     # 10. Recursive Forecasting.
    #     last_row = data.iloc[-1]
    #     start_date = data[datetime_col].iloc[-1]
    #     if self.period_option == 'Hourly':
    #         freq = 'H'
    #     elif self.period_option == 'Daily':
    #         freq = 'D'
    #     elif self.period_option == '5 Minutes':
    #         freq = '5T'
    #     elif self.period_option == 'Monthly':
    #         freq = 'M'
    #     elif self.period_option == 'Yearly':
    #         freq = 'Y'
    #     else:
    #         freq = 'H'
    #     future_dates = pd.date_range(start=start_date, periods=self.forecast_value+1, freq=freq)[1:]

    #     recursive_preds = []
    #     current_features = last_row.drop([datetime_col, target_col]).copy()
    #     step = 0
    #     for dt in future_dates:
    #         next_pred = model.predict(current_features.values.reshape(1, -1))[0]
    #         recursive_preds.append(next_pred)
    #         new_features = current_features.copy()
    #         print("&&&&&&&&&&&&&&&&&&&&&&&+++++++++=new_features===",new_features)
    #         if 'lag_1' in new_features.index and 'lag_2' in new_features.index:
    #             new_features['lag_2'] = current_features['lag_1']
    #             new_features['lag_1'] = next_pred
    #         for feat in exog_feats:
    #             new_features[feat] = exog_forecasts[feat][step]
    #         current_features = new_features.copy()
    #         step += 1

    #     # # 11. Build forecast DataFrame.
    #     # future_df = pd.DataFrame({datetime_col: future_dates})
    #     # future_df["forecast"] = recursive_preds
    #     # actual_range = data[target_col].iloc[-self.forecast_value:].values
    #     # if len(actual_range) != self.forecast_value:
    #     #     actual_range = np.repeat(data[target_col].iloc[-1], self.forecast_value)
    #     # future_df["actual"] = actual_range

    #     # # 12. Final display: only datetime and forecast.
    #     # display_df = future_df[[datetime_col, "forecast"]].copy()
    #     # 11. Build forecast DataFrame.
    #     # 11. Build forecast DataFrame.
    #     future_df = pd.DataFrame({datetime_col: future_dates})
    #     future_df["forecast"] = recursive_preds

    #     # Add forecasted exogenous features
    #     for feat in exog_feats:
    #         future_df[feat] = exog_forecasts[feat]

    #     # Add lag_1 and lag_2 from recursive values (optional)
    #     lag_1_vals = [np.nan] + recursive_preds[:-1]
    #     lag_2_vals = [np.nan, np.nan] + recursive_preds[:-2]
    #     future_df["lag_1"] = lag_1_vals[:self.forecast_value]
    #     future_df["lag_2"] = lag_2_vals[:self.forecast_value]

    #     # Add actuals (if known)
    #     actual_range = data[target_col].iloc[-self.forecast_value:].values
    #     if len(actual_range) != self.forecast_value:
    #         actual_range = np.repeat(data[target_col].iloc[-1], self.forecast_value)
    #     future_df["actual"] = actual_range

    #     # Restore original user-selected features
    #     original_selected_features = [col for col in feature_cols if col != datetime_col]
    #     for col in original_selected_features:
    #         if col not in future_df.columns:
    #             if col in data.columns:
    #                 future_df[col] = data[col].iloc[-1]
    #             else:
    #                 future_df[col] = None

    #     print("+++=========+==original_selected_features=============",original_selected_features)
    #     # Decode categorical columns if encoded
    #     for col, encoder in encoders.items():
    #             if col in future_df.columns:
    #                 try:
    #                     inverse_map = dict(zip(range(len(encoder.classes_)), encoder.classes_))
    #                     future_df[col] = future_df[col].map(inverse_map)
    #                 except Exception as e:
    #                     print(f"Decoding failed for {col}: {e}")
    #     print("+++=========+==original_selected_features=============",original_selected_features)
    #     # Final display: include datetime, forecast, actual, and all original user features
    #     display_cols = [datetime_col, "forecast", "actual"] + original_selected_features
    #     display_cols = list(dict.fromkeys(display_cols))  # Remove duplicates, preserve order
    #     display_df = future_df[display_cols].copy()

    #     preds_train = model.predict(X_train)
    #     mape = mean_absolute_percentage_error(y_train, preds_train)

    #     return display_df, mape



































# class ForecastPipeline:
#     def __init__(
#         self,
#         xgb_params=None,
#         fine_tune_method="None",
#         period_option="Daily",
#         forecast_value=10
#     ):
#         # Default XGBoost params
#         if xgb_params is None:
#             xgb_params = {
#                 "base_score": 0.5,
#                 "booster": "gbtree",
#                 "objective": "reg:squarederror",
#                 "n_estimators": 500,
#                 "learning_rate": 0.01,
#                 "max_depth": 5,
#                 "min_child_weight": 5,
#                 "reg_lambda": 10,
#                 "reg_alpha": 5
#             }
#         self.xgb_params = xgb_params
#         self.fine_tune_method = fine_tune_method
#         self.period_option = period_option
#         self.forecast_value = forecast_value

#     @staticmethod
#     def robust_parse_date(val):
#         s = str(val).strip()
#         if not s:
#             return pd.NaT
#         if s.isdigit() and len(s) == 8:
#             try:
#                 dt = datetime.strptime(s, "%Y%m%d")
#                 if 1900 <= dt.year <= 2100:
#                     return pd.Timestamp(dt)
#             except ValueError:
#                 return pd.NaT
#         if re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", s):
#             return pd.NaT
#         if not any(d in s for d in ["-", "/", "."]):
#             return pd.NaT
#         try:
#             dt = parse(s, fuzzy=False)
#             if 1900 <= dt.year <= 2100:
#                 return pd.Timestamp(dt)
#         except Exception:
#             return pd.NaT
#         return pd.NaT

#     @staticmethod
#     def detect_datetime_column(df, selected_cols, threshold=0.8, exclude_non_date_text=True):
#         for col in selected_cols:
#             if pd.api.types.is_datetime64_any_dtype(df[col]):
#                 return col, 1.0
#         best_col, best_ratio = None, 0.0
#         for col in selected_cols:
#             if exclude_non_date_text:
#                 mixed = df[col].astype(str).str.contains(r"[A-Za-z].*\d|\d.*[A-Za-z]").mean()
#                 if mixed > 0.5:
#                     continue
#             parsed = df[col].apply(ForecastPipeline.robust_parse_date)
#             ratio = parsed.notna().mean()
#             if ratio > best_ratio:
#                 best_ratio, best_col = ratio, col
#         return (best_col, best_ratio) if best_ratio >= threshold else (None, best_ratio)

#     @staticmethod
#     def add_time_features(df, datetime_col):
#         dt = df[datetime_col]
#         df['year'] = dt.dt.year
#         df['month'] = dt.dt.month
#         df['day'] = dt.dt.day
#         df['weekday'] = dt.dt.weekday
#         df['hour'] = dt.dt.hour
#         df['minute'] = dt.dt.minute
#         df['second'] = dt.dt.second
#         df['week_number'] = dt.dt.isocalendar().week.astype(int)
#         df['is_month_start'] = dt.dt.is_month_start.astype(int)
#         df['is_month_end'] = dt.dt.is_month_end.astype(int)
#         df['quarter'] = dt.dt.quarter
#         return df

#     @staticmethod
#     def create_time_features(df, datetime_col, target_col, default_lags=[1, 2], default_rolling=[3, 5]):
#         df = df.copy()
#         df = ForecastPipeline.add_time_features(df, datetime_col)
#         for lag in default_lags:
#             df[f'lag_{lag}'] = df[target_col].shift(lag)
#         for window in default_rolling:
#             df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
#         return df

#     def run(self, df, feature_cols, target_col):
#         # Early exit
#         if self.forecast_value == 0:
#             return pd.DataFrame(columns=feature_cols + ['forecast']), None

#         # 1. Detect datetime
#         datetime_col, ratio = self.detect_datetime_column(df, feature_cols)
#         if datetime_col is None:
#             raise ValueError("Could not detect a datetime column among the selected features.")
#         df = df.copy()
#         df[datetime_col] = df[datetime_col].apply(self.robust_parse_date)
#         global_max = df[datetime_col].max()

#         # 2. Define grouping & additional features
#         group_features = [c for c in feature_cols if c != datetime_col and not pd.api.types.is_numeric_dtype(df[c])]
#         st.write(group_features)
#         print("===================GROUP FEATURE :====;",group_features)
#         additional_features = [c for c in feature_cols if c != datetime_col and c not in group_features]
#         st.write(additional_features)
#         print("======================++++++ ADD FEATURES +++++++++++++++==;", additional_features)

#         # 3. Build group IDs
#         if group_features:
#             join_sym = '__Jos__'
#             df['__group_id__'] = df[group_features].astype(str).agg(join_sym.join, axis=1)
#             groups = df['__group_id__'].unique()
#         else:
#             groups = [None]

#         all_forecasts = []
#         mape_list = []

#         # freq map
#         freq_map = {'hourly':'H','daily':'D','5 minutes':'5T','weekly':'W','monthly':'M','yearly':'Y'}
#         freq = freq_map.get(self.period_option.lower(), 'D')

#         for grp in groups:
#             df_grp = df[df['__group_id__']==grp].copy() if grp is not None else df.copy()
#             df_grp = df_grp.dropna(subset=[datetime_col])
#             df_grp.sort_values(by=datetime_col, inplace=True)
#             if len(df_grp) < 10:
#                 continue
            
#             new_group= group_features+additional_features
#             print("==================++++++",new_group)
#             # Prepare features
#             Xf = df_grp[new_group].copy()
#             cats = Xf.select_dtypes(include=['object']).columns.tolist()
#             # print("=================CATS+++++++++++;",cats)
#             nums = [c for c in Xf if c not in cats]
#             for c in cats:
#                 Xf[c] = LabelEncoder().fit_transform(Xf[c])
#             Xn = pd.DataFrame(StandardScaler().fit_transform(Xf[nums]), columns=nums, index=Xf.index) if nums else pd.DataFrame(index=Xf.index)
#             # print("#$#$#$#$ NUMb====",Xn)
#             Xc = Xf[cats].copy() if cats else pd.DataFrame(index=Xf.index)
#             Xp = pd.concat([Xc, Xn], axis=1)

#             # print("--------------------- XP========",Xp)

#             proc = pd.concat([df_grp[[datetime_col]], Xp, df_grp[[target_col]]], axis=1)
#             # print("--------------------- proc========$$$$$$",proc)
#             proc = self.create_time_features(proc, datetime_col, target_col)
#             # print("--------------------- proc========#####################$$$$$$",proc)
#             proc.dropna(inplace=True)
#             if len(proc) < 10:
#                 continue

#             X_full = proc.drop(columns=[datetime_col, target_col])
#             # print("========++====++++===+++++:X_full",X_full)
#             y_full = proc[target_col]
#             # print("================y_full========++++",y_full)
#             X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
#             model = xgb.XGBRegressor(**self.xgb_params)

#             # tuning
#             search = None
#             pg = {"max_depth":[3,5,7],"learning_rate":[0.01,0.05,0.1],"n_estimators":[100,300,500],"min_child_weight":[1,3,5],"reg_lambda":[0,10,20],"reg_alpha":[0,5,10]}
#             if self.fine_tune_method=="Grid Search": search = GridSearchCV(model, pg, cv=3, n_jobs=-1, verbose=1)
#             elif self.fine_tune_method=="Random Search": search = RandomizedSearchCV(model, pg, cv=3, n_iter=10, n_jobs=-1, verbose=1, random_state=42)
#             elif self.fine_tune_method=="Bayesian Optimization" and BayesSearchCV: search = BayesSearchCV(model, pg, cv=3, n_iter=15, n_jobs=-1, verbose=1, random_state=42)
#             if search: search.fit(X_tr, y_tr); model=search.best_estimator_
#             else: model.fit(X_tr, y_tr, eval_set=[(X_tr,y_tr),(X_te,y_te)], verbose=False)

#             mape_list.append(mean_absolute_percentage_error(y_tr, model.predict(X_tr)))

#             # exogenous
#             exog_feats = [c for c in X_full if c not in ['lag_1','lag_2']]
#             # print("************************* Exog_Feats ******************",exog_feats)
#             exog_fc = {}
            
#             for feat in exog_feats:
#                 s = proc[feat].dropna()
#                 # print("------------------ SSSSS--------------",s)
#                 if len(s)<2: exog_fc[feat]=np.repeat(s.iloc[-1], self.forecast_value)
#                 else:
#                     try: exog_fc[feat]=ARIMA(s,order=(1,1,1)).fit(disp=False).forecast(self.forecast_value)
#                     except: exog_fc[feat]=np.repeat(s.iloc[-1], self.forecast_value)
#             # print("**^*^^*^*^*^*^*^* exog_fc ================",exog_fc)
#             last = proc.iloc[-1].drop([datetime_col, target_col]).copy()
#             fo = []
#             for i, dt in enumerate(pd.date_range(start=global_max, periods=self.forecast_value+1, freq=freq)[1:]):
#                 tf = self.add_time_features(pd.DataFrame({datetime_col:[dt]}), datetime_col).iloc[0].drop(datetime_col)
#                 last.update(tf)
#                 x = last[X_full.columns].values.reshape(1,-1)
#                 print("========================= X_full ##############",x)
#                 p = model.predict(x)[0]
#                 print("==========================PREDICTED===============",p)
#                 # shift
#                 if 'lag_1' in last and 'lag_2' in last: last['lag_2']=last['lag_1']; last['lag_1']=p
#                 for feat in exog_feats: last[feat]=exog_fc[feat][i]
#                 rec = {datetime_col:dt, 'forecast':p}
#                 print("=====++++++++++++ REC ++++++++++++++",rec)
#                 # add group and additional cols
#                 if group_features:
#                     for gf in group_features: rec[gf]=df_grp[gf].iloc[0]
#                 for af in additional_features: rec[af]=df_grp[af].iloc[-1]
#                 fo.append(rec)

#             all_forecasts.extend(fo)

#         if not all_forecasts:
#             raise ValueError("No forecast results generated for any group.")
#         final = pd.DataFrame(all_forecasts)
#         return final, (sum(mape_list)/len(mape_list) if mape_list else None)




























# class ForecastPipeline:
#     def __init__(
#         self,
#         xgb_params=None,
#         fine_tune_method="None",
#         period_option="Daily",
#         forecast_value=10,
#         group_cols=None,
#         min_rows=5,               # ← new: minimum rows after dropna
#     ):
#         # Default XGBoost params
#         if xgb_params is None:
#             xgb_params = {
#                 "base_score": 0.5,
#                 "booster": "gbtree",
#                 "objective": "reg:squarederror",
#                 "n_estimators": 500,
#                 "learning_rate": 0.01,
#                 "max_depth": 5,
#                 "min_child_weight": 5,
#                 "reg_lambda": 10,
#                 "reg_alpha": 5
#             }

#         self.xgb_params      = xgb_params
#         self.fine_tune_method = fine_tune_method
#         self.period_option    = period_option
#         self.forecast_value   = forecast_value
#         self.group_cols       = group_cols
#         self.min_rows         = min_rows

#     # ------------------------------
#     # Utility Functions
#     # ------------------------------
#     @staticmethod
#     def robust_parse_date(val):
#         s = str(val).strip()
#         if not s:
#             return pd.NaT

#         # YYYYMMDD
#         if s.isdigit() and len(s) == 8:
#             try:
#                 dt = datetime.strptime(s, "%Y%m%d")
#                 if 1900 <= dt.year <= 2100:
#                     return dt
#             except ValueError:
#                 return pd.NaT

#         # reject mixed letters+digits
#         if re.search(r"[A-Za-z]", s):
#             return pd.NaT

#         # require delimiter
#         if not any(d in s for d in ("-", "/", ".")):
#             return pd.NaT

#         try:
#             dt = parse(s, fuzzy=False)
#             if 1900 <= dt.year <= 2100:
#                 return dt
#         except Exception:
#             return pd.NaT

#         return pd.NaT

#     @staticmethod
#     def detect_datetime_column(df, selected_cols, threshold=0.8, exclude_non_date_text=True):
#         best_col = None
#         best_ratio = 0.0
#         for col in selected_cols:
#             if exclude_non_date_text:
#                 alpha_num = df[col].astype(str).str.contains(r"[A-Za-z].*\d|\d.*[A-Za-z]").mean()
#                 if alpha_num > 0.5:
#                     continue
#             if not pd.api.types.is_object_dtype(df[col]):
#                 continue
#             parsed = df[col].apply(ForecastPipeline.robust_parse_date)
#             ratio = parsed.notnull().mean()
#             if ratio > best_ratio:
#                 best_ratio = ratio
#                 best_col = col
#         return (best_col, best_ratio) if best_ratio >= threshold else (None, best_ratio)

#     @staticmethod
#     def add_time_features(df, datetime_col):
#         # df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
#         df["year"]           = df[datetime_col].dt.year
#         df["month"]          = df[datetime_col].dt.month
#         df["day"]            = df[datetime_col].dt.day
#         df["weekday"]        = df[datetime_col].dt.weekday
#         df["hour"]           = df[datetime_col].dt.hour
#         df["minute"]         = df[datetime_col].dt.minute
#         df["second"]         = df[datetime_col].dt.second
#         df["week_number"]    = df[datetime_col].dt.isocalendar().week.astype(int)
#         df["is_month_start"] = df[datetime_col].dt.is_month_start.astype(int)
#         df["is_month_end"]   = df[datetime_col].dt.is_month_end.astype(int)
#         df["quarter"]        = df[datetime_col].dt.quarter
#         return df

#     @staticmethod
#     def create_time_features(df, datetime_col, target_col, default_lags=[1, 2], default_rolling=[3, 5]):
#         df = df.copy()
#         df = ForecastPipeline.add_time_features(df, datetime_col)
#         # df = ForecastPipeline.add_time_features(df, datetime_col)
#         for lag in default_lags:
#             df[f"lag_{lag}"] = df[target_col].shift(lag)
#         for window in default_rolling:
#             df[f"rolling_mean_{window}"] = df[target_col].rolling(window=window).mean()
#         return df

#     # ------------------------------
#     # Main Pipeline Method
#     # ------------------------------
#     def run(self, df, feature_cols, target_col):
#         # 1. Detect datetime column
#         datetime_col, best_ratio = self.detect_datetime_column(df, feature_cols)
#         if datetime_col is None:
#             raise ValueError("Could not detect a datetime column among the selected features.")
#         df[datetime_col] = df[datetime_col].apply(self.robust_parse_date)
#         global_max_date = pd.to_datetime(df[datetime_col], errors="coerce").max()

#         # 2. Determine grouping columns
#         if self.group_cols:
#             group_features = [
#                 c for c in self.group_cols
#                 if c in feature_cols and c != datetime_col
#             ]
#         else:
#             group_features = [
#                 col for col in feature_cols
#                 if col != datetime_col and not pd.api.types.is_numeric_dtype(df[col])
#             ]

#         # 3. Additional input features
#         additional_features = [
#             col for col in feature_cols
#             if col not in group_features + [datetime_col]
#         ]

#         # 4. Prepare groups
#         if group_features:
#             groups = df.groupby(group_features)
#         else:
#             groups = [(None, df)]

#         forecast_results = []
#         mape_list = []

#         # 5. Loop over each group
#         for grp_vals, df_grp in groups:
#             # Build a dict so we can re-attach group values later
#             if group_features:
#                 if len(group_features) == 1:
#                     grp_dict = {group_features[0]: grp_vals}
#                 else:
#                     grp_dict = dict(zip(group_features, grp_vals))
#             else:
#                 grp_dict = {}

#             df_grp = df_grp.copy()
#             df_grp[datetime_col] = df_grp[datetime_col].apply(self.robust_parse_date)
#             if df_grp[datetime_col].isnull().any():
#                 continue
#             df_grp.sort_values(by=datetime_col, inplace=True)

#             # 6. Prepare X_features
#             X_features = df_grp[additional_features].copy() if additional_features else pd.DataFrame(index=df_grp.index)

#             # 6a. Drop zero‑variance numeric columns
#             num_cols = X_features.select_dtypes(include='number').columns.tolist()
#             zero_var = [c for c in num_cols if X_features[c].nunique(dropna=False) <= 1]
#             if zero_var:
#                 X_features.drop(columns=zero_var, inplace=True)
#                 num_cols = [c for c in num_cols if c not in zero_var]

#             # 6b. Encode categoricals
#             cat_cols = X_features.select_dtypes(include=['object']).columns.tolist()
#             for col in cat_cols:
#                 le = LabelEncoder()
#                 X_features[col] = le.fit_transform(X_features[col])

#             # 6c. Scale numerics
#             if num_cols:
#                 scaler = StandardScaler()
#                 X_scaled = scaler.fit_transform(X_features[num_cols])
#                 X_num = pd.DataFrame(X_scaled, columns=num_cols, index=X_features.index)
#             else:
#                 X_num = pd.DataFrame(index=X_features.index)

#             X_cat = X_features[cat_cols].copy() if cat_cols else pd.DataFrame(index=X_features.index)

#             if not X_cat.empty and not X_num.empty:
#                 X_proc = pd.concat([X_cat, X_num], axis=1)
#             elif not X_cat.empty:
#                 X_proc = X_cat.copy()
#             else:
#                 X_proc = X_num.copy()

#             # 7. Combine datetime, features, and target; then create time features
#             df_proc = pd.concat([df_grp[[datetime_col]], X_proc, df_grp[[target_col]]], axis=1)
#             df_proc = self.create_time_features(df_proc, datetime_col, target_col)
#             df_proc.dropna(inplace=True)

#             # 7b. Skip small groups
#             if len(df_proc) < self.min_rows:
#                 continue

#             X_full = df_proc.drop(columns=[datetime_col, target_col])
#             print("Features used for training:", X_full.columns.tolist())
#             y_full = df_proc[target_col]

#             # 8. Train/test split & model training
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_full, y_full, test_size=0.2, random_state=42
#             )
#             model = xgb.XGBRegressor(**self.xgb_params)
#             model.fit(
#                 X_train, y_train,
#                 eval_set=[(X_train, y_train), (X_test, y_test)],
#                 verbose=False
#             )

#             # 9. Record MAPE
#             preds_train = model.predict(X_train)
#             group_mape  = mean_absolute_percentage_error(y_train, preds_train)
#             mape_list.append(group_mape)

#             # 10. Recursive forecasting
#             freq_map = {
#                 "5 Minutes": "5T",
#                 "Hourly":    "H",
#                 "Daily":     "D",
#                 "Monthly":   "M",
#                 "Yearly":    "Y"
#             }
#             freq = freq_map.get(self.period_option, "D")
#             future_dates = pd.date_range(
#                 start=global_max_date,
#                 periods=self.forecast_value + 1,
#                 freq=freq
#             )[1:]

#             recursive_preds = []
#             last_row       = df_proc.iloc[-1]
#             current_feats  = last_row.drop([datetime_col, target_col]).copy()

#             # Forecast exogenous features via ARIMA
#             exog_feats = [c for c in X_full.columns if c not in ["lag_1", "lag_2"]]
#             exog_forecasts = {}
#             for feat in exog_feats:
#                 try:
#                     series = df_proc[feat].dropna()
#                     arima_model = ARIMA(series, order=(1,1,1)).fit()
#                     fc = arima_model.forecast(steps=self.forecast_value)
#                     exog_forecasts[feat] = fc.values
#                 except Exception:
#                     exog_forecasts[feat] = np.repeat(df_proc[feat].iloc[-1], self.forecast_value)

#             for step, dt in enumerate(future_dates):
#                 print("=========------ current fity =-==-=-=-=",current_feats)
#                 next_pred = model.predict(current_feats.values.reshape(1, -1))[0]
#                 recursive_preds.append(next_pred)

#                 new_feats = current_feats.copy()
#                 if "lag_1" in new_feats.index and "lag_2" in new_feats.index:
#                     new_feats["lag_2"] = current_feats["lag_1"]
#                     new_feats["lag_1"] = next_pred

#                 for feat in exog_feats:
#                     new_feats[feat] = exog_forecasts[feat][step]

#                 current_feats = new_feats

#             # 11. Build forecast DataFrame
#             future_df = pd.DataFrame({
#                 datetime_col: future_dates,
#                 "forecast":   recursive_preds
#             })
#             # re-attach group cols & carry-forward additional features
#             for col, val in grp_dict.items():
#                 future_df[col] = val
#             for col in additional_features:
#                 future_df[col] = df_grp[col].iloc[-1]

#             forecast_results.append(future_df)

#         # 12. Combine all group forecasts
#         if not forecast_results:
#             raise ValueError("No forecast results generated for any group.")

#         final_forecast_df = pd.concat(forecast_results, ignore_index=True)
#         overall_mape      = sum(mape_list) / len(mape_list) if mape_list else None

#         return final_forecast_df, overall_mape






































# try:
#     from skopt import BayesSearchCV
# except ImportError:
#     BayesSearchCV = None


# class ForecastPipeline:
#     def __init__(
#         self,
#         xgb_params=None,
#         fine_tune_method="None",
#         period_option="Daily",
#         forecast_value=10,
#         model_path="model.joblib",
#         scaler_path="scaler.joblib",
#         encoder_dir="encoders"
#     ):
#         # Default XGBoost params
#         self.xgb_params = xgb_params or {
#             "base_score": 0.5,
#             "booster": "gbtree",
#             "objective": "reg:squarederror",
#             "n_estimators": 500,
#             "learning_rate": 0.01,
#             "max_depth": 5,
#             "min_child_weight": 5,
#             "reg_lambda": 10,
#             "reg_alpha": 5
#         }
#         self.fine_tune_method = fine_tune_method
#         self.period_option = period_option.lower()  # normalize
#         self.forecast_value = forecast_value

#         # Paths for persisting artifacts
#         self.model_path = model_path
#         self.scaler_path = scaler_path
#         self.encoder_dir = encoder_dir

#         # Will be filled in during .fit()
#         self.encoders = {}
#         self.scaler = None
#         self.model = None

#     @staticmethod
#     def robust_parse_date(val):
#         s = str(val).strip()
#         if not s:
#             return pd.NaT
#         # YYYYMMDD
#         if s.isdigit() and len(s) == 8:
#             try:
#                 dt = datetime.strptime(s, "%Y%m%d")
#                 if 1900 <= dt.year <= 2100:
#                     return dt
#             except ValueError:
#                 return pd.NaT
#         # reject any letters
#         if re.search(r"[A-Za-z]", s):
#             return pd.NaT
#         # require delimiter
#         if not any(d in s for d in ("-", "/", ".")):
#             return pd.NaT
#         try:
#             dt = parse(s, fuzzy=False)
#             if 1900 <= dt.year <= 2100:
#                 return dt
#         except Exception:
#             return pd.NaT
#         return pd.NaT

#     @staticmethod
#     def detect_datetime_column(df, cols, threshold=0.8, exclude_non_date_text=True):
#         # 1) If any column is already datetime dtype, choose that
#         for c in cols:
#             if pd.api.types.is_datetime64_any_dtype(df[c]):
#                 return c, 1.0

#         # 2) Otherwise try parsing object/string columns
#         best_col, best_ratio = None, 0.0
#         for c in cols:
#             if exclude_non_date_text:
#                 mixed = (
#                     df[c]
#                     .astype(str)
#                     .str.contains(r"[A-Za-z].*\d|\d.*[A-Za-z]")
#                     .mean()
#                 )
#                 if mixed > 0.5:
#                     continue
#             if not (
#                 pd.api.types.is_object_dtype(df[c])
#                 or pd.api.types.is_string_dtype(df[c])
#             ):
#                 continue
#             parsed = df[c].apply(ForecastPipeline.robust_parse_date)
#             ratio = parsed.notna().mean()
#             if ratio > best_ratio:
#                 best_ratio, best_col = ratio, c

#         if best_ratio >= threshold:
#             return best_col, best_ratio
#         return None, best_ratio

#     @staticmethod
#     def add_time_features(df, datetime_col):
#         df[datetime_col] = pd.to_datetime(df[datetime_col])
#         dt = df[datetime_col]
#         df["tf_year"]           = dt.dt.year
#         df["tf_month"]          = dt.dt.month
#         df["tf_day"]            = dt.dt.day
#         df["tf_weekday"]        = dt.dt.weekday
#         df["tf_hour"]           = dt.dt.hour
#         df["tf_minute"]         = dt.dt.minute
#         df["tf_second"]         = dt.dt.second
#         df["tf_week_number"]    = dt.dt.isocalendar().week.astype(int)
#         df["tf_is_month_start"] = dt.dt.is_month_start.astype(int)
#         df["tf_is_month_end"]   = dt.dt.is_month_end.astype(int)
#         df["tf_quarter"]        = dt.dt.quarter
#         return df

#     @staticmethod
#     def create_time_features(
#         df, datetime_col, target_col,
#         default_lags=(1, 2), default_rolling=(3, 5)
#     ):
#         df = ForecastPipeline.add_time_features(df, datetime_col)
#         for lag in default_lags:
#             df[f"lag_{lag}"] = df[target_col].shift(lag)
#         for w in default_rolling:
#             df[f"rol_{w}"] = df[target_col].rolling(window=w).mean()
#         return df

#     def fit(self, df, feature_cols, target_col):
#         # 1) Detect datetime column
#         dt_col, ratio = self.detect_datetime_column(df, df.columns)
#         if dt_col is None:
#             raise ValueError("Could not detect a datetime column in your DataFrame.")

#         # 2) **Remove datetime from features to avoid duplicates**
#         feature_cols = [c for c in feature_cols if c != dt_col]

#         # 3) Parse & coerce to datetime64
#         df = df.copy()
#         df[dt_col] = df[dt_col].apply(self.robust_parse_date)
#         df[dt_col] = pd.to_datetime(df[dt_col])
#         df.dropna(subset=[dt_col], inplace=True)
#         df.sort_values(dt_col, inplace=True)

#         # 4) Split X / y
#         X = df[feature_cols].copy()
#         y = df[target_col]

#         # 5) Label‑encode categoricals
#         cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
#         os.makedirs(self.encoder_dir, exist_ok=True)
#         for c in cats:
#             le = LabelEncoder().fit(X[c].astype(str))
#             X[c] = le.transform(X[c].astype(str))
#             self.encoders[c] = le
#             joblib.dump(le, os.path.join(self.encoder_dir, f"{c}.joblib"))

#         # 6) Build time‑features + lags/rolling, drop NAs
#         df_proc = pd.concat([df[[dt_col]], X, y], axis=1)
#         df_proc = self.create_time_features(df_proc, dt_col, target_col)
#         df_proc.dropna(inplace=True)

#         X_full = df_proc.drop(columns=[dt_col, target_col])
#         y_full = df_proc[target_col]

#         # 7) Fit & save scaler
#         self.scaler = StandardScaler().fit(X_full)
#         self.feature_names_after_fit = X_full.columns.tolist() # Store feature names
#         joblib.dump(self.scaler, self.scaler_path)
#         X_scaled = pd.DataFrame(
#             self.scaler.transform(X_full),
#             columns=X_full.columns, index=X_full.index
#         )

#         # 8) Train/test split
#         X_tr, X_te, y_tr, y_te = train_test_split(
#             X_scaled, y_full, test_size=0.2, random_state=42
#         )

#         # 9) Train model (with optional tuning)
#         model = xgb.XGBRegressor(**self.xgb_params)
#         param_grid = {
#             "max_depth": [3, 5, 7],
#             "learning_rate": [0.01, 0.05, 0.1],
#             "n_estimators": [100, 300, 500],
#             "min_child_weight": [1, 3, 5],
#             "reg_lambda": [0, 10, 20],
#             "reg_alpha": [0, 5, 10],
#         }
#         search = None
#         if self.fine_tune_method == "Grid Search":
#             search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
#         elif self.fine_tune_method == "Random Search":
#             search = RandomizedSearchCV(
#                 model, param_grid, cv=3, n_iter=10,
#                 n_jobs=-1, verbose=1, random_state=42
#             )
#         elif self.fine_tune_method == "Bayesian Optimization" and BayesSearchCV:
#             search = BayesSearchCV(
#                 model, param_grid, cv=3, n_iter=15,
#                 n_jobs=-1, verbose=1, random_state=42
#             )

#         if search:
#             search.fit(X_tr, y_tr)
#             model = search.best_estimator_
#         else:
#             model.fit(
#                 X_tr, y_tr,
#                 eval_set=[(X_tr, y_tr), (X_te, y_te)],
#                 verbose=True
#             )

#         self.model = model
#         joblib.dump(model, self.model_path)

#         return mean_absolute_percentage_error(y_tr, model.predict(X_tr))

#     def _build_future_frame(self, last_proc, datetime_col, feature_cols, target_col):
#         freq_map = {
#             "hourly": "H", "daily": "D", "5 minutes": "5T",
#             "weekly": "W", "monthly": "M", "yearly": "Y",
#         }
#         freq = freq_map.get(self.period_option, "D")
#         last_date = last_proc[datetime_col].max()
#         future_dates = pd.date_range(
#             start=last_date,
#             periods=self.forecast_value + 1,
#             freq=freq
#         )[1:]

#         fut = pd.DataFrame({datetime_col: future_dates})
#         fut = self.add_time_features(fut, datetime_col)

#         # Constant categorical codes
#         for c in self.encoders:
#             if c in feature_cols:
#                 fut[c] = last_proc[c].iloc[-1]

#         # Lags & rolling
#         for lag in (1, 2):
#             fut[f"lag_{lag}"] = last_proc[target_col].shift(lag).iloc[-1]
#         for w in (3, 5):
#             fut[f"rol_{w}"] = last_proc[target_col].rolling(w).mean().iloc[-1]

#         # Drop date, dedupe, scale, re‑attach date
#         X_fut = fut.drop(columns=[datetime_col])

#         # Select and order columns based on training
#         X_fut = X_fut[self.feature_names_after_fit]

#         X_scaled = pd.DataFrame(
#             self.scaler.transform(X_fut),
#             columns=X_fut.columns, index=X_fut.index
#         )
#         X_scaled[datetime_col] = fut[datetime_col]
#         return X_scaled

#     def predict(self, df, feature_cols, target_col):
#         # 1) Load artifacts if needed
#         if self.model is None:
#             self.model = joblib.load(self.model_path)
#         if self.scaler is None:
#             self.scaler = joblib.load(self.scaler_path)
#         for fn in os.listdir(self.encoder_dir):
#             col = fn.replace(".joblib", "")
#             self.encoders[col] = joblib.load(os.path.join(self.encoder_dir, fn))

#         # 2) Detect datetime column
#         dt_col, _ = self.detect_datetime_column(df, df.columns)
#         if dt_col is None:
#             raise ValueError("Could not detect a datetime column in your DataFrame.")

#         # 3) Remove it from features
#         feature_cols = [c for c in feature_cols if c != dt_col]

#         # 4) Parse & sort
#         df = df.copy()
#         df[dt_col] = df[dt_col].apply(self.robust_parse_date)
#         df[dt_col] = pd.to_datetime(df[dt_col])
#         df.dropna(subset=[dt_col], inplace=True)
#         df.sort_values(dt_col, inplace=True)

#         # 5) Re‑encode categoricals
#         for c, le in self.encoders.items():
#             if c in feature_cols:
#                 df[c] = le.transform(df[c].astype(str))

#         # 6) Build processed history
#         df_proc = pd.concat([df[[dt_col]], df[feature_cols], df[[target_col]]], axis=1)
#         df_proc = self.create_time_features(df_proc, dt_col, target_col)
#         df_proc.dropna(inplace=True)

#         cats = [c for c in feature_cols if c in self.encoders]
#         results = []

#         # 7) Group → build future → predict
#         for group_vals, grp in df_proc.groupby(cats or [None]):
#             if grp.shape[0] < max(5, self.forecast_value):
#                 continue

#             X_fut = self._build_future_frame(
#                 last_proc=grp,
#                 datetime_col=dt_col,
#                 feature_cols=feature_cols,
#                 target_col=target_col
#             )
#             dates = X_fut[dt_col]
#             preds = self.model.predict(X_fut.drop(columns=[dt_col]))

#             base = {}
#             if cats:
#                 vals = group_vals if isinstance(group_vals, tuple) else (group_vals,)
#                 base = {c: v for c, v in zip(cats, vals)}

#             out = pd.DataFrame({
#                 **base,
#                 dt_col: dates,
#                 "forecast": preds
#             })
#             results.append(out)

#         return pd.concat(results, ignore_index=True)

#     def run(self, df, feature_cols, target_col):
#         mape = self.fit(df, feature_cols, target_col)
#         fc   = self.predict(df, feature_cols, target_col)
#         return fc, mape

































 
 
 
 






# ## Optional Bayesian optimization
# try:
#     from skopt import BayesSearchCV
# except ImportError:
#     BayesSearchCV = None


# class ForecastPipeline:
#     def __init__(
#         self,
#         xgb_params=None,
#         fine_tune_method="None",
#         period_option="Daily",
#         forecast_value=10
#     ):
#         # Default XGBoost params
#         if xgb_params is None:
#             xgb_params = {
#                 "base_score": 0.5,
#                 "booster": "gbtree",
#                 "objective": "reg:squarederror",
#                 "n_estimators": 500,
#                 "learning_rate": 0.01,
#                 "max_depth": 5,
#                 "min_child_weight": 5,
#                 "reg_lambda": 10,
#                 "reg_alpha": 5
#             }
#         self.xgb_params = xgb_params
#         self.fine_tune_method = fine_tune_method
#         self.period_option = period_option
#         self.forecast_value = forecast_value

#     @staticmethod
#     def robust_parse_date(val):
#         s = str(val).strip()
#         if not s:
#             return pd.NaT

#         # YYYYMMDD
#         if s.isdigit() and len(s) == 8:
#             try:
#                 dt = datetime.strptime(s, "%Y%m%d")
#                 if 1900 <= dt.year <= 2100:
#                     return pd.Timestamp(dt)
#             except ValueError:
#                 return pd.NaT

#         # reject mixed alphanumeric
#         if re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", s):
#             return pd.NaT

#         # need a delimiter
#         if not any(d in s for d in ["-", "/", "."]):
#             return pd.NaT

#         try:
#             dt = parse(s, fuzzy=False)
#             if 1900 <= dt.year <= 2100:
#                 return pd.Timestamp(dt)
#         except Exception:
#             return pd.NaT

#         return pd.NaT

#     @staticmethod
#     def detect_datetime_column(df, selected_cols, threshold=0.8, exclude_non_date_text=True):
#         # Accept existing datetime dtype
#         for col in selected_cols:
#             if pd.api.types.is_datetime64_any_dtype(df[col]):
#                 return col, 1.0

#         best_col, best_ratio = None, 0.0
#         for col in selected_cols:
#             if exclude_non_date_text:
#                 mixed = df[col].astype(str).str.contains(r"[A-Za-z].*\d|\d.*[A-Za-z]").mean()
#                 if mixed > 0.5:
#                     continue
#             parsed = df[col].apply(ForecastPipeline.robust_parse_date)
#             ratio = parsed.notna().mean()
#             if ratio > best_ratio:
#                 best_ratio, best_col = ratio, col

#         return (best_col, best_ratio) if best_ratio >= threshold else (None, best_ratio)

#     @staticmethod
#     def add_time_features(df, datetime_col):
#         dt = df[datetime_col]
#         df['year'] = dt.dt.year
#         df['month'] = dt.dt.month
#         df['day'] = dt.dt.day
#         df['weekday'] = dt.dt.weekday
#         df['hour'] = dt.dt.hour
#         df['minute'] = dt.dt.minute
#         df['second'] = dt.dt.second
#         df['week_number'] = dt.dt.isocalendar().week.astype(int)
#         df['is_month_start'] = dt.dt.is_month_start.astype(int)
#         df['is_month_end'] = dt.dt.is_month_end.astype(int)
#         df['quarter'] = dt.dt.quarter
#         return df

#     @staticmethod
#     def create_time_features(df, datetime_col, target_col, default_lags=[1, 2], default_rolling=[3, 5]):
#         df = ForecastPipeline.add_time_features(df, datetime_col)
#         for lag in default_lags:
#             df[f'lag_{lag}'] = df[target_col].shift(lag)
#         for window in default_rolling:
#             df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
#         return df

#     def run(self, df, feature_cols, target_col):
#         # Early exit if no forecast requested
#         if self.forecast_value == 0:
#             return pd.DataFrame(columns=[target_col, 'forecast']), None

#         # 1. Detect datetime column
#         datetime_col, ratio = self.detect_datetime_column(df, feature_cols)
#         if datetime_col is None:
#             raise ValueError("Could not detect a datetime column among the selected features.")

#         # parse datetime and get global max
#         df = df.copy()
#         df[datetime_col] = df[datetime_col].apply(self.robust_parse_date)
#         global_max_date = df[datetime_col].max()

#         # 2. All non-datetime selected columns are features
#         additional_features = [c for c in feature_cols if c != datetime_col]

#         # 3. Prepare data for modeling
#         df = df.dropna(subset=[datetime_col])
#         df.sort_values(by=datetime_col, inplace=True)

#         # Extract X and y
#         X = df[additional_features].copy()
#         cats = X.select_dtypes(include=['object']).columns.tolist()
#         nums = [c for c in X.columns if c not in cats]

#         for c in cats:
#             X[c] = LabelEncoder().fit_transform(X[c])
#         if nums:
#             X_num = pd.DataFrame(
#                 StandardScaler().fit_transform(X[nums]),
#                 columns=nums, index=X.index
#             )
#         else:
#             X_num = pd.DataFrame(index=X.index)
#         X_cat = X[cats].copy() if cats else pd.DataFrame(index=X.index)
#         X_proc = pd.concat([X_cat, X_num], axis=1)

#         df_proc = pd.concat([df[[datetime_col]], X_proc, df[[target_col]]], axis=1)
#         df_proc = self.create_time_features(df_proc, datetime_col, target_col)
#         df_proc.dropna(inplace=True)
#         if len(df_proc) < 10:
#             raise ValueError("Not enough data after processing. Need at least 10 rows.")

#         X_full = df_proc.drop(columns=[datetime_col, target_col])
#         y_full = df_proc[target_col]

#         # 4. Train/test split
#         X_tr, X_te, y_tr, y_te = train_test_split(
#             X_full, y_full, test_size=0.2, random_state=42
#         )
#         model = xgb.XGBRegressor(**self.xgb_params)

#         # 5. Fine tuning
#         search = None
#         param_grid = {
#             "max_depth": [3,5,7],
#             "learning_rate": [0.01,0.05,0.1],
#             "n_estimators": [100,300,500],
#             "min_child_weight": [1,3,5],
#             "reg_lambda": [0,10,20],
#             "reg_alpha": [0,5,10]
#         }
#         if self.fine_tune_method == "Grid Search":
#             search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
#         elif self.fine_tune_method == "Random Search":
#             search = RandomizedSearchCV(
#                 model, param_grid, cv=3, n_iter=10,
#                 n_jobs=-1, verbose=1, random_state=42
#             )
#         elif self.fine_tune_method == "Bayesian Optimization" and BayesSearchCV:
#             search = BayesSearchCV(
#                 model, param_grid, cv=3, n_iter=15,
#                 n_jobs=-1, verbose=1, random_state=42
#             )

#         if search:
#             search.fit(X_tr, y_tr)
#             model = search.best_estimator_
#         else:
#             model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_te, y_te)], verbose=True)

#         # 6. Compute training MAPE
#         mape_train = mean_absolute_percentage_error(y_tr, model.predict(X_tr))

#         # 7. Recursive forecasting
#         freq_map = {
#             'hourly': 'H', 'daily': 'D', '5 minutes': '5T',
#             'weekly': 'W', 'monthly': 'M', 'yearly': 'Y'
#         }
#         freq = freq_map.get(self.period_option.lower(), 'D')
#         future_dates = pd.date_range(
#             start=global_max_date,
#             periods=self.forecast_value+1,
#             freq=freq
#         )[1:]

#         last_row = df_proc.iloc[-1].drop([datetime_col, target_col])
#         feature_order = X_full.columns.tolist()

#         # exogenous forecasts
#         exog_feats = [c for c in feature_order if c not in ['lag_1','lag_2']]
#         exog_forecasts = {}
#         for feat in exog_feats:
#             series = df_proc[feat].dropna()
#             if len(series) < 2:
#                 exog_forecasts[feat] = np.repeat(series.iloc[-1], self.forecast_value)
#             else:
#                 try:
#                     ar = ARIMA(series, order=(1,1,1)).fit(disp=False)
#                     exog_forecasts[feat] = ar.forecast(steps=self.forecast_value)
#                 except Exception:
#                     exog_forecasts[feat] = np.repeat(series.iloc[-1], self.forecast_value)

#         forecasts = []
#         current = last_row.copy()
#         for step, dt in enumerate(future_dates):
#             st.write(dt)
#             st.write(current)
#             try:
#                 # update time features
#                 tmp = pd.DataFrame({datetime_col: [dt]})
#                 tf = self.add_time_features(tmp, datetime_col).iloc[0].drop(datetime_col)
#                 current.update(tf)
#                 # predict
#                 x = current[feature_order].values.reshape(1, -1)
#                 pred = model.predict(x)[0]
#                 # shift lags
#                 if 'lag_1' in current and 'lag_2' in current:
#                     current['lag_2'] = current['lag_1']
#                     current['lag_1'] = pred
#                 # update exog
#                 for feat in exog_feats:
#                     current[feat] = exog_forecasts[feat][step]
#             except Exception as ex:
#                 raise RuntimeError(f"Error forecasting step {step} (date {dt}): {ex}")
#             forecasts.append({'datetime': dt, 'forecast': pred})

#         # 8. Build final DataFrame
#         final_df = pd.DataFrame(forecasts)
#         return final_df, mape_train

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
# # For Bayesian optimization (if needed)
# try:
#     from skopt import BayesSearchCV
# except ImportError:
#     BayesSearchCV = None
 
 
# class ForecastPipeline:
#     def __init__(self, xgb_params=None, fine_tune_method="None", period_option="Daily", forecast_value=10):
#         # If no XGBoost parameters are provided, use default values.
#         if xgb_params is None:
#             xgb_params = {
#                 "base_score": 0.5,
#                 "booster": "gbtree",
#                 "objective": "reg:squarederror",
#                 "n_estimators": 500,
#                 "learning_rate": 0.01,
#                 "max_depth": 5,
#                 "min_child_weight": 5,
#                 "reg_lambda": 10,
#                 "reg_alpha": 5
#             }
#         self.xgb_params = xgb_params
#         self.fine_tune_method = fine_tune_method
#         self.period_option = period_option
#         self.forecast_value = forecast_value
 
#     # ------------------------------
#     # Utility Functions
#     # ------------------------------
#     @staticmethod
#     def robust_parse_date(val):
#         """
#         Try to parse a value as a date.
#         - If the value is an 8-digit number, assume YYYYMMDD.
#         - Otherwise, require that the string contains at least one typical date delimiter ('-', '/', or '.'),
#         and does NOT contain letters mixed with numbers.
#         - If these conditions are not met, return NaT.
#         Returns a datetime if parsing succeeds and the year is plausible; otherwise returns pd.NaT.
#         """
#         s = str(val).strip()
#         if not s:
#             return pd.NaT
 
#         # If the string is all digits and exactly 8 characters, assume YYYYMMDD.
#         if s.isdigit() and len(s) == 8:
#             try:
#                 dt = datetime.strptime(s, '%Y%m%d')
#                 if 1900 <= dt.year <= 2100:
#                     return dt
#             except ValueError:
#                 return pd.NaT
 
#         # If it contains any letters, do not attempt parsing.
#         if re.search(r"[A-Za-z]", s):
#             return pd.NaT
 
#         # For non-digit-only strings, require a delimiter (-, /, .).
#         if not any(delim in s for delim in ['-', '/', '.']):
#             return pd.NaT
 
#         try:
#             dt = parse(s, fuzzy=False)  # Set fuzzy=False to avoid extracting parts of strings
#             if 1900 <= dt.year <= 2100:
#                 return dt
#         except Exception:
#             return pd.NaT
 
#         return pd.NaT
 
#     @staticmethod
#     def detect_datetime_column(df, selected_cols, threshold=0.8,exclude_non_date_text=True):
#         best_col = None
#         best_ratio = 0
#         for col in selected_cols:
#             # Exclude columns with non-date-like alphanumeric values
#             if exclude_non_date_text:
#                 alphanumeric_ratio = df[col].astype(str).str.contains(r"[A-Za-z].*\d|\d.*[A-Za-z]").mean()
#                 if alphanumeric_ratio > 0.5:  # If more than 50% contain mixed text/numbers, ignore
#                     continue
 
#             if not pd.api.types.is_object_dtype(df[col]):
#                 continue
#             parsed = df[col].apply(ForecastPipeline.robust_parse_date)
#             ratio = parsed.notnull().mean()
#             if ratio > best_ratio:
#                 best_ratio = ratio
#                 best_col = col
#         return (best_col, best_ratio) if best_ratio >= threshold else (None, best_ratio)
 
#     @staticmethod
#     def add_time_features(df, datetime_col):
#         df['year'] = df[datetime_col].dt.year
#         df['month'] = df[datetime_col].dt.month
#         df['day'] = df[datetime_col].dt.day
#         df['weekday'] = df[datetime_col].dt.weekday
#         df['hour'] = df[datetime_col].dt.hour
#         df['minute'] = df[datetime_col].dt.minute
#         df['second'] = df[datetime_col].dt.second
#         df['week_number'] = df[datetime_col].dt.isocalendar().week.astype(int)
#         df['is_month_start'] = df[datetime_col].dt.is_month_start.astype(int)
#         df['is_month_end'] = df[datetime_col].dt.is_month_end.astype(int)
#         df['quarter'] = df[datetime_col].dt.quarter
#         return df
 
#     @staticmethod
#     def create_time_features(df, datetime_col, target_col, default_lags=[1, 2], default_rolling=[3, 5]):
#         df = ForecastPipeline.add_time_features(df, datetime_col)
#         for lag in default_lags:
#             df[f'lag_{lag}'] = df[target_col].shift(lag)
#         for window in default_rolling:
#             df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
#         return df
 
 
#     def run(self, df, feature_cols, target_col):
#         # 1. Detect the datetime column.
#         datetime_col, best_ratio = self.detect_datetime_column(df, feature_cols)
#         if datetime_col is None:
#             raise ValueError("Could not detect a datetime column among the selected features.")
 
#         # Compute the global maximum date available in the file.
#         global_max_date = pd.to_datetime(df[datetime_col], errors="coerce").max()
 
#         # 2. Define grouping features: only non-numeric (excluding the datetime column).
#         group_features = [col for col in feature_cols if col != datetime_col and not pd.api.types.is_numeric_dtype(df[col])]
        
#         # 3. Define additional (input) features: numeric ones and any extra non‑grouping columns.
#         additional_features = [col for col in feature_cols if col != datetime_col and col not in group_features]
 
#         # 4. Create composite group id if grouping features exist.
#         if group_features:
#             join_on_symbol = '__Jos__'
#             group_id_col = join_on_symbol.join(group_features)
#             df[group_id_col] = df[group_features].astype(str).agg(join_on_symbol.join, axis=1)
#             unique_groups = df[group_id_col].unique()
#         else:
#             unique_groups = [None]
#             group_id_col = None
 
#         forecast_results_list = []
#         mape_list = []  # Collect MAPE for each group
 
#         # 5. Loop over each group.
#         for grp in unique_groups:
#             if grp is not None:
#                 df_grp = df[df[group_id_col] == grp].copy()
#             else:
#                 df_grp = df.copy()
 
#             # Ensure the datetime column is parsed and sorted.
#             df_grp[datetime_col] = df_grp[datetime_col].apply(self.robust_parse_date)
#             if df_grp[datetime_col].isnull().any():
#                 continue  # Skip groups with date conversion issues.
#             df_grp.sort_values(by=datetime_col, inplace=True)
 
#             # 6. Process additional features.
#             X_features = df_grp[additional_features].copy() if additional_features else pd.DataFrame()
#             cat_cols = X_features.select_dtypes(include=['object']).columns.tolist()
#             num_cols = [col for col in X_features.columns if col not in cat_cols]
            
#             # Label encode categorical features.
#             for col in cat_cols:
#                 le = LabelEncoder()
#                 X_features[col] = le.fit_transform(X_features[col])
            
#             # Scale numeric columns.
#             if num_cols:
#                 scaler = StandardScaler()
#                 X_features_scaled_num = pd.DataFrame(scaler.fit_transform(X_features[num_cols]),
#                                                     columns=num_cols, index=X_features.index)
#             else:
#                 X_features_scaled_num = pd.DataFrame(index=X_features.index)
            
#             X_features_scaled_cat = X_features[cat_cols].copy() if cat_cols else pd.DataFrame(index=X_features.index)
            
#             if not X_features_scaled_cat.empty and not X_features_scaled_num.empty:
#                 X_features_processed = pd.concat([X_features_scaled_cat, X_features_scaled_num], axis=1)
#             elif not X_features_scaled_cat.empty:
#                 X_features_processed = X_features_scaled_cat.copy()
#             else:
#                 X_features_processed = X_features_scaled_num.copy()
 
#             # 7. Combine datetime, processed features, and target.
#             df_proc = pd.concat([df_grp[[datetime_col]], X_features_processed, df_grp[[target_col]]], axis=1)
#             df_proc = self.create_time_features(df_proc, datetime_col, target_col)
#             df_proc.dropna(inplace=True)
 
#             # Check if there is sufficient data.
#             if len(df_proc) < 10:
#                 continue
 
#             X_full = df_proc.drop(columns=[datetime_col, target_col])
#             y_full = df_proc[target_col]
 
#             # 8. Train-test split and train the XGBoost model.
#             X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
#             model = xgb.XGBRegressor(**self.xgb_params)
 
#             # Fine tuning functionality.
#             if self.fine_tune_method != "None":
#                 param_grid = {
#                     "max_depth": [3, 5, 7],
#                     "learning_rate": [0.01, 0.05, 0.1],
#                     "n_estimators": [100, 300, 500],
#                     "min_child_weight": [1, 3, 5],
#                     "reg_lambda": [0, 10, 20],
#                     "reg_alpha": [0, 5, 10]
#                 }
#                 if self.fine_tune_method == "Grid Search":
#                     # st.write("Performing Grid Search...")
#                     search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
#                 elif self.fine_tune_method == "Random Search":
#                     # st.write("Performing Random Search...")
#                     search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=10, n_jobs=-1, verbose=1, random_state=42)
#                 elif self.fine_tune_method == "Bayesian Optimization":
#                     # st.write("Performing Bayesian Optimization...")
#                     # Uncomment and adjust the next line if you have scikit-optimize installed:
#                     # search = BayesSearchCV(model, param_grid, cv=3, n_iter=15, n_jobs=-1, verbose=1, random_state=42)
#                     search = None  # Remove this line once you enable Bayesian Optimization.
#                 else:
#                     search = None
 
#                 if search is not None:
#                     search.fit(X_train, y_train)
#                     model = search.best_estimator_
#                 else:
#                     model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
#             else:
#                 model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
 
#             # Compute MAPE on the training data.
#             preds_train = model.predict(X_train)
#             group_mape = mean_absolute_percentage_error(y_train, preds_train)
#             mape_list.append(group_mape)
 
#             # 9. Recursive Forecasting using the global max date.
#             start_date = global_max_date
#             freq = {'Hourly': 'H', 'Daily': 'D', '5 Minutes': '5T', 'weekly': 'W', 'Monthly': 'M', 'Yearly': 'Y'}.get(self.period_option, 'H')
#             future_dates = pd.date_range(start=start_date, periods=self.forecast_value+1, freq=freq)[1:]
 
#             recursive_preds = []
#             # Use the last row's features as a starting point.
#             last_row = df_proc.iloc[-1]
#             current_features = last_row.drop([datetime_col, target_col]).copy()
 
#             # Forecast exogenous features with ARIMA.
#             exog_feats = [col for col in X_full.columns if col not in ['lag_1', 'lag_2']]
#             exog_forecasts = {}
#             for feat in exog_feats:
#                 try:
#                     series = df_proc[feat].dropna()
#                     arima_model = ARIMA(series, order=(1,1,1)).fit()
#                     forecast_series = arima_model.forecast(steps=self.forecast_value)
#                     exog_forecasts[feat] = forecast_series.values
#                 except Exception as ex:
#                     exog_forecasts[feat] = np.repeat(df_proc[feat].iloc[-1], self.forecast_value)
 
#             step = 0
#             for dt in future_dates:
#                 st.write(dt)
#                 st.write(current_features)
#                 next_pred = model.predict(current_features.values.reshape(1, -1))[0]
#                 recursive_preds.append(next_pred)
#                 new_features = current_features.copy()
#                 if 'lag_1' in new_features.index and 'lag_2' in new_features.index:
#                     new_features['lag_2'] = current_features['lag_1']
#                     new_features['lag_1'] = next_pred
#                 for feat in exog_feats:
#                     new_features[feat] = exog_forecasts[feat][step]
#                 current_features = new_features.copy()
#                 step += 1
 
#             # 10. Build forecast DataFrame for this group.
#             future_df = pd.DataFrame({datetime_col: future_dates, "forecast": recursive_preds})
            
#             # Include grouping columns in the output.
#             if group_features:
#                 for gf in group_features:
#                     future_df[gf] = df_grp[gf].iloc[0]
            
#             # Also include additional (numeric) input columns (using last observed values).
#             if additional_features:
#                 for col in additional_features:
#                     future_df[col] = df_grp[col].iloc[-1]
            
#             # Optionally, add group-level MAPE to each row.
#             # future_df["MAPE"] = group_mape
 
#             forecast_results_list.append(future_df)
 
#         # 11. Combine all group forecasts.
#         if forecast_results_list:
#             final_forecast_df = pd.concat(forecast_results_list, axis=0)
#             overall_mape = sum(mape_list) / len(mape_list) if mape_list else None
#             return final_forecast_df, group_mape
#         else:
#             raise ValueError("No forecast results generated for any group.")
 
 