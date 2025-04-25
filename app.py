import streamlit as st
import pandas as pd
from streamlit_forecasting import autots_run_pipeline,ForecastPipeline
 
# Set wide layout for better UI
st.set_page_config(layout="wide")
 
# Title and description
st.title("Forecasting with AutoTS and XGBoost")
st.write("This app is used to forecast time series data using AutoTS and XGBoost models.")
 
with st.sidebar:
    st.header("Model Selection")
    model_options = ["None", "AutoTS", "XGBoost"]
    selected_model = st.radio("Select Model:", model_options, key="model_name")
 
    if "last_model" not in st.session_state:
        st.session_state.last_model = selected_model
    
    if selected_model != st.session_state.last_model:
        st.session_state.last_model = selected_model
        st.rerun()
 
# File Upload Section
st.subheader("Upload your dataset (CSV or Excel)")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
 
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.session_state.df = df
    st.session_state.columns = df.columns.tolist()
    st.write("Data Preview:")
    st.dataframe(df)
 
if "df" in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    columns = st.session_state.columns
 
    if selected_model == "AutoTS":
        st.subheader("AutoTS Model Customization Configuration")
        con_1, con_2, = st.columns([1, 1])
 
        with con_1:
            prediction_interval = st.selectbox(
                "Prediction Interval:",
                [0.9, 0.99, 0.95, 0.85, 0.8],
                help=(
                    "**🔧 Fine-Tuning Impact:**\n"
                    "- Adjusts the confidence range of predictions.\n\n"
                    "**📈 Benefit:**\n"
                    "- **Higher values (e.g., 0.99):** Wider range, more uncertainty.\n"
                    "- **Lower values (e.g., 0.8):** Narrower range, more confident but riskier.\n\n"
                    "**⚙️ How it Works:**\n"
                    "- The model calculates predictions within the given confidence level to estimate uncertainty."
                )
            )
 
            model_list = st.selectbox(
                "Model List:",
                ["superfast", "fast", "all"],
                help=(
                    "**🔧 Fine-Tuning Impact:**\n"
                    "- Defines the speed and complexity of model selection.\n\n"
                    "**📈 Benefit:**\n"
                    "- **'superfast':** Fastest, uses ARIMA & Prophet.\n"
                    "- **'fast':** Balances speed and accuracy.\n"
                    "- **'all':** Includes all models (most accurate but slowest).\n\n"
                    "**⚙️ How it Works:**\n"
                    "- Determines which forecasting models AutoTS will use, affecting training time and accuracy."
                )
            )
 
            max_generations = st.selectbox(
                "Max Generations:",
                [1, 5, 10, 20],
                help=(
                    "**🔧 Fine-Tuning Impact:**\n"
                    "- Controls the number of optimization cycles for model selection.\n\n"
                    "**📈 Benefit:**\n"
                    "- **Higher values (e.g., 10-20):** More tuning, better accuracy.\n"
                    "- **Lower values (e.g., 1-5):** Faster training but may reduce accuracy.\n\n"
                    "**⚙️ How it Works:**\n"
                    "- AutoTS runs multiple generations to refine the best-performing models."
                )
            )
 
        with con_2:
            ensemble = st.selectbox(
                "Ensemble Strategy:",
                ["auto", "simple", "horizontal-max", "horizontal-min", "horizontal-median", "dist"],
                help=(
                    "**🔧 Fine-Tuning Impact:**\n"
                    "- Defines how multiple models are combined for better predictions.\n\n"
                    "**📈 Benefit:**\n"
                    "- **'auto':** Automatically selects the best approach.\n"
                    "- **'simple':** Averages all models.\n"
                    "- **'horizontal-max':** Uses the highest forecast.\n"
                    "- **'horizontal-min':** Uses the lowest forecast.\n"
                    "- **'horizontal-median':** Uses the median prediction.\n"
                    "- **'dist':** Weighted averaging based on model confidence.\n\n"
                    "**⚙️ How it Works:**\n"
                    "- Combines outputs from multiple models to reduce errors and improve accuracy."
                )
            )
 
            transformer_list = st.selectbox(
                "Transformer List:",
                ["fast", "superfast", "scalable"],
                help=(
                    "**🔧 Fine-Tuning Impact:**\n"
                    "- Defines how input data is preprocessed before training.\n\n"
                    "**📈 Benefit:**\n"
                    "- **'fast':** Minimal transformations for speed.\n"
                    "- **'superfast':** Even faster transformations.\n"
                    "- **'scalable':** Includes methods suited for large datasets.\n\n"
                    "**⚙️ How it Works:**\n"
                    "- Preprocessing normalizes, scales, or transforms data to improve model performance."
                )
            )
 
            num_validations = st.selectbox(
                "Num Validations:",
                [1, 3, 5, 10],
                help=(
                    "**🔧 Fine-Tuning Impact:**\n"
                    "- Determines how many cross-validation tests are performed.\n\n"
                    "**📈 Benefit:**\n"
                    "- **More validations (e.g., 5-10):** More reliable results, but slower.\n"
                    "- **Fewer validations (e.g., 1-3):** Faster training but may risk overfitting.\n\n"
                    "**⚙️ How it Works:**\n"
                    "- Splits historical data multiple times for better model evaluation."
                )
            )
 
        st.subheader("Forecasting Configuration")
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
 
        with col1:
            selected_feature = st.multiselect("Feature of Data:", columns)
        with col2:
            selected_target = st.selectbox("Target of Data:", columns)
        with col3:
            frequency = st.selectbox("Forecast Frequency:", ["5Min","Day", "Week", "Month"])
        with col4:
            forecast_length = st.number_input("Forecast Period:", min_value=1, step=1)
 
        if st.button("Submit"):
            if selected_feature and selected_target and forecast_length:
                frequency_mapping = {"5Min":"5T","Day": "D", "Week": "W", "Month": "M"}
                frequency = frequency_mapping.get(frequency, "D")
                autots_config = {
                    "prediction_interval": prediction_interval,
                    "ensemble": ensemble,
                    "model_list": model_list,
                    "transformer_list": transformer_list,
                    "max_generations": max_generations,
                    "num_validations": num_validations,
                }
                
                # st.write(autots_config)
                with st.spinner("Processing for forecasting ... "):
                    result,mape_score = autots_run_pipeline(df, selected_feature, selected_target, forecast_length, frequency,autots_config)
                    if result is not None and mape_score is not None:
                        st.dataframe(result)
                        st.write(f"MAPE Score: {mape_score:.4f}")
                        csv = result.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Forecast CSV", data=csv, file_name='forecast_results.csv', mime='text/csv')
                    else:
                        print("According to the data your Forecasting period is to more (let's try to do less)")
                        st.write("According to the data your Forecasting period is to more (let's try to do less)")
            else:
                st.warning("Please select both Feature and Target columns!")
    
    elif selected_model == "XGBoost":
        st.subheader("XGBoost Model Configuration")
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

        with col1:
            selected_features = st.multiselect(
                "Select features (exclude datetime—you'll get that auto‑detected):",
                columns
            )
        with col2:
            selected_target = st.selectbox("Select target column:", columns)
        with col3:
            freq = st.selectbox(
                "Forecast frequency:",
                ["5 minutes", "hourly", "daily", "monthly", "yearly"]
            )
        with col4:
            forecast_length = st.number_input(
                "Forecast horizon (# of periods):",
                min_value=1, step=1
            )

        st.subheader("XGBoost Hyperparameters")
        tun1, tun2, tun3, tun4, tun5 = st.columns(5)
        tun6, tun7, tun8, tun9       = st.columns(4)

        with tun1:
            base_score = st.number_input("base_score", value=0.5, step=0.1)
        with tun2:
            booster = st.selectbox("booster", ["gbtree", "dart", "gblinear"])
        with tun3:
            objective = st.selectbox("objective", ["reg:squarederror"])
        with tun4:
            n_estimators = st.number_input("n_estimators", value=500, step=50)
        with tun5:
            learning_rate = st.number_input(
                "learning_rate", value=0.01, step=0.01, format="%.2f"
            )
        with tun6:
            max_depth = st.number_input("max_depth", value=5, step=1)
        with tun7:
            min_child_weight = st.number_input("min_child_weight", value=5, step=1)
        with tun8:
            reg_lambda = st.number_input("reg_lambda", value=10, step=1)
        with tun9:
            reg_alpha = st.number_input("reg_alpha", value=5, step=1)

        xgb_params = {
            "base_score": base_score,
            "booster": booster,
            "objective": objective,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha
        }

        st.subheader("Fine‑tuning method")
        fine_tune_method = st.selectbox(
            "Select method", ["None", "Grid Search", "Random Search", "Bayesian Optimization"]
        )

        if st.button("Submit"):
            with st.spinner("Training & forecasting… this may take a minute"):
                # try:
                    # normalize the freq string for our pipeline
                period_option = freq.lower()  # e.g. "5 minutes", "hourly", etc.

                pipeline = ForecastPipeline(
                        xgb_params=xgb_params,
                        fine_tune_method=fine_tune_method,
                        period_option=period_option,
                        forecast_value=forecast_length
                    )
                results, mape = pipeline.run(df, selected_features, selected_target)

                # except Exception as e:
                #     st.error(f"❌ Error: {e}")
                #     st.stop()

            st.subheader("Forecast Results")
            st.dataframe(results)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV", data=csv,
                file_name="forecast_results.csv", mime="text/csv"
            )

            st.markdown(f"**Train MAPE:** {mape:.3%}")








    # elif selected_model == "XGBoost":
    #     st.subheader("XGBoost Model Configuration")
    #     col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
 
    #     with col1:
    #         selected_feature = st.multiselect("Feature of Data:", columns)
    #     with col2:
    #         selected_target = st.selectbox("Target of Data:", columns)
    #     with col3:
    #         frequency = st.selectbox("Forecast Frequency:", ["5 Minutes","Hourly","Daily","Monthly","Yearly"])
    #     with col4:
    #         forecast_length = st.number_input("Forecast Period:", min_value=1, step=1)
        
    #     st.subheader("XGBoost Model Parameters")
    #     st.write("Basic best parameters for XGBoost (which may depend on datasets).")
    #     tun1, tun2, tun3, tun4,tun5 = st.columns([1, 1, 1, 1,1])
    #     tun6, tun7, tun8, tun9 = st.columns([1, 1, 1,1])
    #     with tun1:
    #         base_score = st.number_input("Base Score", value=0.5, step=0.1)
    #     with tun2:
    #         booster = st.selectbox("Booster", options=["gbtree", "dart", "gblinear"], index=0)
    #     with tun3:
    #         objective = st.selectbox("Objective",options=["reg:squarederror"])
    #     with tun4:
    #         n_estimators = st.number_input("n_estimators", value=500, step=50)
    #     with tun5:
    #         learning_rate = st.number_input("Learning Rate", value=0.01, step=0.01, format="%.2f")
    #     with tun6:
    #         max_depth = st.number_input("Max Depth", value=5, step=1)
    #     with tun7:
    #         min_child_weight = st.number_input("Min Child Weight", value=5, step=1)
    #     with tun8:
    #         reg_lambda = st.number_input("Reg Lambda", value=10, step=1)
    #     with tun9:
    #         reg_alpha = st.number_input("Reg Alpha", value=5, step=1)
    #     xgb_params = {
    #         "base_score": base_score,
    #         "booster": booster,
    #         "objective": objective,
    #         "n_estimators": n_estimators,
    #         "learning_rate": learning_rate,
    #         "max_depth": max_depth,
    #         "min_child_weight": min_child_weight,
    #         "reg_lambda": reg_lambda,
    #         "reg_alpha": reg_alpha
    #     }
    #     st.subheader("Model Fine Tuning")
    #     fine_tune_method = st.selectbox("Select Fine Tuning Method", ["None", "Grid Search", "Random Search", "Bayesian Optimization"])
 
    #     if st.button("Submit"):
    #         with st.spinner("Please wait a minute for the forecasting results...."):
    #             try:
    #                 pipeline = ForecastPipeline(xgb_params=xgb_params,
    #                                             fine_tune_method=fine_tune_method,
    #                                             period_option=frequency,
    #                                             forecast_value=forecast_length)
    #                 results, mape = pipeline.run(df, selected_feature, selected_target)
    #             except Exception as e:
    #                 st.error(f"Error during pipeline execution: {e}")
    #                 st.stop()
    #         st.subheader("Forecast Results")
    #         st.dataframe(results)
    #         csv = results.to_csv(index=False).encode("utf-8")
    #         st.download_button("Download Forecast CSV", data=csv, file_name="forecast_results.csv", mime="text/csv")
    #         st.write("MAPE:", mape)