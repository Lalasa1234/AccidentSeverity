# The file to be imported as module should be in .py format
from prediction import model_prediction
import streamlit as st
import joblib
import xgboost
import numpy as np

model_path = 'ModelBuilding_Notebook/XGBoost_model.joblib'
model_loaded = joblib.load(model_path)

st.title("Accident Severity Prediction App")

#creating option list for dropdown menu
options_day = ["Monday", 'Sunday',"Friday","Wednesday",  "Saturday","Thursday","Tuesday" ]
options_age = ['18-30', '31-50', 'Under 18','Over 51']
options_driver_exp = ['1-2yr','Above 10yr','5-10yr', '2-5yr', 'No Licence', 'Below 1yr' ]
options_vehicle_type = ['Automobile', 'Public (> 45 seats)','Lorry (41-100Q)', 'Public (13-45 seats)',
'Lorry (11-40Q)','Long lorry','Public (12 seats)','Taxi',
'Pick up upto 10Q','Stationwagen', 'Special vehicle','Other', 'Motorcycle']
options_area = ['Residential areas','Office areas','  Recreational areas',' Industrial areas','Other',
' Church areas','  Market areas','Rural village areas',' Outside rural areas',' Hospital areas','School areas']
options_junc = ['No junction','Y Shape','Crossing','O Shape','Other','T Shape']

with st.form('Form'):
    st.header('Set the specifications of day, driver, vehicle and location conditions')
    Hour = st.number_input (label='Hour: ',min_value = 0, max_value = 23, step = 1)
    Min = st.number_input (label='Minute: ',min_value = 0, max_value = 59, step = 1)
    Day_of_week = st.selectbox ('Day of week: ',options = options_day)
    Age_band_of_driver = st.selectbox('Age band of driver: ', options = options_age )
    Driving_experience = st.selectbox('Driving experience: ', options = options_driver_exp )
    Type_of_vehicle	 = st.selectbox('Type of vehicle: ',options = options_vehicle_type)
    Area_accident_occured = st.selectbox('Area: ', options = options_area)
    Types_of_Junction = st.selectbox('Type of junction: ',options = options_junc)

    submit_values = st.form_submit_button ('Predict')

if submit_values:
    Day_of_week = 1 + options_day.index(Day_of_week)
    Age_band_of_driver = 1+ options_age.index(Age_band_of_driver)
    Driving_experience = 1 + options_driver_exp.index(Driving_experience)
    Type_of_vehicle = 1 + options_vehicle_type.index(Type_of_vehicle)
    Area_accident_occured = 1 + options_area.index(Area_accident_occured)
    Types_of_Junction = 1 + options_junc.index(Types_of_Junction)

    data = np.array([Hour, Min, Day_of_week,
       Age_band_of_driver, Driving_experience, Type_of_vehicle,
       Area_accident_occured, Types_of_Junction]).reshape(1,-1)

    prediction = model_prediction(model_loaded,data)

    if prediction[0] == 0:
        value = 'Accident is fatally severe'
    elif prediction[0] == 1:
        value = 'Accident is seriously severe'
    else:
        value = 'Accident is slightly severe'

    st.header('Here is the prediction: ')
    st.success(f'{value}')
