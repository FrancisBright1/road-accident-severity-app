import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Road Traffic Accident Analysis",
    page_icon="🚗",
    layout="wide",
)

st.title("Road Traffic Accident Analysis and Severity Prediction")
st.subheader("Python and Streamlit Industrial Training Project")
st.caption(
    "Synthetic academic dataset only. Not official statistics. "
    "Not for real emergency or safety decisions."
)

@st.cache_data
def load_data():
    df = pd.read_csv("road_traffic_accidents.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month_Name"] = df["Date"].dt.month_name()
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/accident_severity_model.pkl")

def unique_options(series):
    return sorted(series.dropna().astype(str).unique().tolist())

def speed_group(speed):
    if speed < 40:
        return "Low"
    if speed < 80:
        return "Medium"
    return "High"

def experience_group(years):
    if years <= 2:
        return "Novice"
    if years <= 10:
        return "Intermediate"
    return "Experienced"

def vehicle_count_group(n):
    if n == 1:
        return "Single"
    if n == 2:
        return "Two"
    return "Multiple"

data = load_data()

st.sidebar.header("Analytics Filters")
st.sidebar.caption("These filters affect Analytics and Data Explorer, not the prediction form.")

state_options = ["All"] + unique_options(data["State"])
year_options = ["All"] + sorted(data["Year"].dropna().unique().tolist())
severity_options = ["All", "Minor", "Serious", "Fatal"]
vehicle_options = ["All"] + unique_options(data["Vehicle_Type"])
weather_options = ["All"] + unique_options(data["Weather_Condition"])
road_options = ["All"] + unique_options(data["Road_Condition"])
tod_options = ["All", "Early Morning", "Morning", "Afternoon", "Evening", "Night"]

selected_state = st.sidebar.selectbox("State", state_options)
selected_year = st.sidebar.selectbox("Year", year_options)
selected_severity = st.sidebar.selectbox("Accident Severity", severity_options)
selected_vehicle = st.sidebar.selectbox("Vehicle Type", vehicle_options)
selected_weather = st.sidebar.selectbox("Weather Condition", weather_options)
selected_road = st.sidebar.selectbox("Road Condition", road_options)
selected_tod = st.sidebar.selectbox("Time of Day", tod_options)

filtered_data = data.copy()
if selected_state != "All":
    filtered_data = filtered_data[filtered_data["State"] == selected_state]
if selected_year != "All":
    filtered_data = filtered_data[filtered_data["Year"] == selected_year]
if selected_severity != "All":
    filtered_data = filtered_data[filtered_data["Accident_Severity"] == selected_severity]
if selected_vehicle != "All":
    filtered_data = filtered_data[filtered_data["Vehicle_Type"] == selected_vehicle]
if selected_weather != "All":
    filtered_data = filtered_data[filtered_data["Weather_Condition"] == selected_weather]
if selected_road != "All":
    filtered_data = filtered_data[filtered_data["Road_Condition"] == selected_road]
if selected_tod != "All":
    filtered_data = filtered_data[filtered_data["Time_of_Day"] == selected_tod]

tab_analytics, tab_predict, tab_explore, tab_info = st.tabs(
    ["Accident Analytics", "Severity Prediction", "Data Explorer", "Model Information"]
)

with tab_analytics:
    st.header("Key Performance Indicators")
    st.caption(f"Showing {len(filtered_data)} of {len(data)} accidents after filters.")

    if len(filtered_data) == 0:
        st.warning("No records match the selected filters. Change a sidebar filter and try again.")
    else:
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        c1.metric("Total Accidents", int(len(filtered_data)))
        c2.metric("Total Casualties", int(filtered_data["Casualties"].sum()))
        c3.metric("Total Fatalities", int(filtered_data["Fatalities"].sum()))
        c4.metric("Total Injuries", int(filtered_data["Injuries"].sum()))
        c5.metric("Average Speed (km/h)", round(filtered_data["Speed_Estimate_kmh"].mean(), 1))
        c6.metric("Average Response (min)", round(filtered_data["Emergency_Response_Minutes"].mean(), 1))

        st.header("Charts")
        left, right = st.columns(2)
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        tod_order = ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]

        with left:
            st.subheader("Accident Severity")
            st.bar_chart(filtered_data["Accident_Severity"].value_counts())
            st.subheader("Accidents by Month")
            st.bar_chart(filtered_data["Month_Name"].value_counts().reindex(month_order).fillna(0))
            st.subheader("Accidents by Vehicle Type")
            st.bar_chart(filtered_data["Vehicle_Type"].value_counts())
            st.subheader("Weather Analysis")
            st.bar_chart(filtered_data["Weather_Condition"].value_counts())

        with right:
            st.subheader("Accidents by State")
            st.bar_chart(filtered_data["State"].value_counts())
            st.subheader("Accidents by Cause")
            st.bar_chart(filtered_data["Cause_of_Accident"].value_counts())
            st.subheader("Road Condition Analysis")
            st.bar_chart(filtered_data["Road_Condition"].value_counts())
            st.subheader("Time-of-Day Analysis")
            st.bar_chart(filtered_data["Time_of_Day"].value_counts().reindex(tod_order).fillna(0))

with tab_predict:
    st.header("Accident Severity Prediction")
    st.info("Fill the form, then click Predict Accident Severity.")

    f1, f2, f3 = st.columns(3)
    with f1:
        p_state = st.selectbox("State", unique_options(data["State"]))
        p_location = st.selectbox("Location", unique_options(data["Location"]))
        p_road_type = st.selectbox("Road Type", unique_options(data["Road_Type"]))
        p_weather = st.selectbox("Weather Condition", unique_options(data["Weather_Condition"]))
        p_road_cond = st.selectbox("Road Condition", unique_options(data["Road_Condition"]))
        p_tod = st.selectbox("Time of Day", ["Early Morning", "Morning", "Afternoon", "Evening", "Night"])
        p_hour = st.slider("Hour of Day", 0, 23, 14)

    with f2:
        p_vehicle = st.selectbox("Vehicle Type", unique_options(data["Vehicle_Type"]))
        p_nveh = st.number_input("Number of Vehicles", min_value=1, max_value=6, value=2)
        p_speed = st.slider("Estimated Speed (km/h)", 20, 140, 60)
        p_age = st.number_input("Driver Age", min_value=18, max_value=70, value=35)
        p_gender = st.radio("Driver Gender", ["Male", "Female"])
        p_exp = st.number_input("Driver Experience (years)", min_value=1, max_value=40, value=8)
        p_cause = st.selectbox("Cause of Accident", unique_options(data["Cause_of_Accident"]))

    with f3:
        p_density = st.selectbox("Traffic Density", ["Low", "Moderate", "High"])
        p_resp = st.slider("Emergency Response Time (minutes)", 5, 90, 20)
        p_vis = st.selectbox("Visibility", ["Good", "Moderate", "Poor"])
        p_seat = st.selectbox("Seatbelt Usage", ["Yes", "No", "Unknown"])
        p_helmet = st.selectbox("Helmet Usage", ["Yes", "No", "Not Applicable"])
        p_alcohol = st.selectbox("Alcohol Involvement", ["Yes", "No", "Unknown"])
        p_tctrl = st.selectbox("Traffic Control", unique_options(data["Traffic_Control"]))
        p_damage = st.selectbox("Property Damage Level", ["Low", "Moderate", "High", "Severe"])
        p_wrel = st.radio("Weather Related", ["Yes", "No"])
        p_cat = st.selectbox("Accident Category", unique_options(data["Accident_Category"]))

    predict_clicked = st.button("Predict Accident Severity")

    if predict_clicked:
        try:
            model = load_model()
            max_exp = max(1, int(p_age) - 17)
            exp_value = min(int(p_exp), max_exp)

            input_row = pd.DataFrame([{
                "State": p_state,
                "Location": p_location,
                "Road_Type": p_road_type,
                "Weather_Condition": p_weather,
                "Road_Condition": p_road_cond,
                "Time_of_Day": p_tod,
                "Vehicle_Type": p_vehicle,
                "Number_of_Vehicles": int(p_nveh),
                "Speed_Estimate_kmh": int(p_speed),
                "Driver_Age": int(p_age),
                "Driver_Gender": p_gender,
                "Driver_Experience_Years": exp_value,
                "Cause_of_Accident": p_cause,
                "Traffic_Density": p_density,
                "Emergency_Response_Minutes": int(p_resp),
                "Visibility": p_vis,
                "Seatbelt_Usage": p_seat,
                "Helmet_Usage": p_helmet,
                "Alcohol_Involvement": p_alcohol,
                "Traffic_Control": p_tctrl,
                "Property_Damage_Level": p_damage,
                "Weather_Related": p_wrel,
                "Accident_Category": p_cat,
                "Year": 2024,
                "Month": 6,
                "Month_Name": "June",
                "Day_Name": "Monday",
                "Hour": int(p_hour),
                "Is_Night": int(p_tod in ["Night", "Early Morning"]),
                "Speed_Category": speed_group(int(p_speed)),
                "Driver_Experience_Category": experience_group(exp_value),
                "Vehicle_Count_Category": vehicle_count_group(int(p_nveh)),
            }])

            expected_cols = list(model.feature_names_in_)
            input_row = input_row.reindex(columns=expected_cols)

            prediction = model.predict(input_row)[0]
            proba = model.predict_proba(input_row)[0]
            classes = model.classes_

            st.markdown("### Prediction Result")
            st.success(f"Predicted Accident Severity: **{str(prediction).upper()}**")

            st.markdown("### Prediction Probabilities")
            for cls, p in zip(classes, proba):
                st.write(f"{cls}: {p * 100:.1f}%")
                st.progress(float(p))

            st.caption(
                "Based on a synthetic academic dataset. "
                "Do not use this output for real emergency decisions."
            )
        except FileNotFoundError:
            st.error("Model file not found at models/accident_severity_model.pkl")
        except Exception as e:
            st.error("Prediction failed. Check that the model file matches this form.")
            st.exception(e)

with tab_explore:
    st.header("Data Explorer")
    st.write("View the records that match the sidebar filters.")
    st.info(f"Filtered records: {len(filtered_data)}")

    search_text = st.text_input("Search in State, Location, Vehicle Type or Cause")

    explorer_data = filtered_data.copy()
    if search_text.strip():
        q = search_text.strip().lower()
        mask = (
            explorer_data["State"].astype(str).str.lower().str.contains(q, na=False)
            | explorer_data["Location"].astype(str).str.lower().str.contains(q, na=False)
            | explorer_data["Vehicle_Type"].astype(str).str.lower().str.contains(q, na=False)
            | explorer_data["Cause_of_Accident"].astype(str).str.lower().str.contains(q, na=False)
        )
        explorer_data = explorer_data[mask]
        st.caption(f"Records after search: {len(explorer_data)}")

    st.dataframe(explorer_data, use_container_width=True)

    csv_bytes = explorer_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered records as CSV",
        data=csv_bytes,
        file_name="filtered_accidents.csv",
        mime="text/csv",
    )

with tab_info:
    st.header("Model Information")
    st.write(
        """
        - Target variable: Accident_Severity (Minor, Serious, Fatal)
        - Training tool: Scikit-learn Pipeline
        - Saved file: models/accident_severity_model.pkl
        - Identifier excluded: Accident_ID
        - Outcome columns excluded from prediction: Casualties, Fatalities, Injuries
        - This is a university demonstration system, not an official road-safety tool.
        """
    )

st.markdown("---")
st.caption("Industrial Training Project | Timi-Tech Digital Consult| Franalysis Insights| University of Uyo ")
