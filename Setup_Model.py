import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import joblib


def get_entities_for_disease(disease="diabetes"):
    if disease == "diabetes":
        cat_entities = ["Gender", "Diabetes"]
        num_entities = ["Delta0", "Delta2"]
        # actual database entries
        # cat_entities = ["Diagnoses - ICD10", "Sex", "Tobacco smoking"]
        # num_entities = ["year of birth", "Glucose", "Body mass index (BMI)", "Glycated haemoglobin (HbA1c)"]

    if disease == "CHD":
        # actual database entries
        # cat_entities = ["alcohol"]
        # num_entities = ["sbp", "tobacco", "ldl", "age", "obesity"]
        cat_entities = ["Gender", "Diabetes"]
        num_entities = ["Jitter_rel", "Jitter_abs"]
    return cat_entities, num_entities


def convert_to_features(df, enc, categorical_columns: list = ['Sex', 'Tobacco smoking']):
    assert isinstance(categorical_columns, list)

    onehot = enc.transform(df[categorical_columns])
    df_onehot = pd.DataFrame(onehot.toarray(), columns=enc.get_feature_names_out(categorical_columns))

    # Concatenate numerical columns with one-hot encoded columns
    columns = list(df.columns)
    for col in categorical_columns:
        columns.remove(col)
    df = pd.concat([df_onehot, df[columns]], axis=1)

    return df

def save_model(model, encoder=None, scaler=None, disease="diabetes"):
    joblib.dump(model, f'{disease}_prediction_model.pkl')
    if encoder:
        joblib.dump(encoder, f'{disease}encoder.pkl')
    joblib.dump(scaler, f'{disease}scaler.pkl')


def load_model(target_disease: str = "diabetes", encoder=True):
    model = joblib.load(f'{target_disease}_prediction_model.pkl')
    scaler = joblib.load(f'{target_disease}scaler.pkl')
    if encoder:
        encoder = joblib.load(f'{target_disease}encoder.pkl')
        return model, encoder, scaler
    else:
        return model, None, scaler


def get_variable_map(target_disease: str = "diabetes"):
    smoking_mapping = {
        "not current": "Ex-smoker",
        "former": "Ex-smoker",
        "ever": "Occasionally",
        "current": "Smokes on most or all days",
        "No Info": "Prefer not to answer",
        "never": "Never smoked"
    }
    if target_disease == "diabetes":
        variable_map = {
            "HbA1c_level": "Glycated haemoglobin (HbA1c)",
            "bmi": "Body mass index (BMI)",
            "blood_glucose_level": "Glucose",
            # "heart_disease": "Diagnoses - ICD10)",
            "gender": "Sex"
        }
    else:
        variable_map = {
            "sbp": "systolic blood pressure automated reading",
            # yearly tobacco use in kg to cigarettes per day
            # "tobacco": "Amount of tobacco currently smoked",
            # yearly alcohol intake(guessing grams/day)? to never, monthly or less, 2 to 4 times a week
            # "alcohol": "Frequency of drinking alcohol",
            "obesity": "Body mass index (BMI)",
            "ldl": "LDL direct",
        }

    return variable_map, smoking_mapping


def train_risk_score_model(target_disease: str = "diabetes", categorical_columns: list = ['Sex', 'Tobacco smoking'],
                           drop_columns: list = ["age", "smoking_history"]):
    # data = pd.read_csv(f'../../examples/{target_disease}_prediction_dataset.csv')
    data = pd.read_csv(f'examples/{target_disease}_prediction_dataset.csv')
    variable_mapping = get_variable_map(target_disease)[0]
    smoking_mapping = get_variable_map(target_disease)[1]
    # year of birth was determined in 2008
    data['Year of birth'] = 2008 - data['age']

    # not current, former, ever,  current, No Info, never to Ex-smoker 2x, Occasionally, Smokes on most or all days,
    # Prefer not to answer, Never smoked
    if target_disease == "diabetes":
        data["Tobacco smoking"] = data["smoking_history"].map(smoking_mapping)

    # Rename variables based on the mapping dictionary
    data = data.rename(columns=variable_mapping)

    # Determine categories of categorical columns
    if target_disease == "diabetes":
        category_names = {}
        for col in categorical_columns:
            category_names[col] = data[col].unique()
        encoder = OneHotEncoder(categories=[category_names[col] for col in categorical_columns])

    scaler = StandardScaler()
    model = LogisticRegression()
    if target_disease == "diabetes":
        # One-hot encode categorical columns
        encoder.fit(data[categorical_columns])
        data = convert_to_features(data, encoder, categorical_columns)

    # Split data into features (X) and target (y)
    y = data[target_disease]
    x = data.drop([target_disease] + drop_columns, axis=1)

    # Scale the input features
    x_scaled = scaler.fit_transform(x)

    # Train the logistic regression model using the scaled data
    model.fit(x_scaled, y)

    # Print the model coefficients
    coef_dict = {}
    for coef, feat in zip(model.coef_[0], x.columns):
        coef_dict[feat] = coef
    print(coef_dict)
    # Evaluate model accuracy
    y_pred = model.predict(x_scaled)
    accuracy = accuracy_score(y, y_pred)
    print('Accuracy:', accuracy)

    if target_disease == "CHD":
        save_model(model, None, scaler, target_disease)
    else:
        save_model(model, encoder, scaler, target_disease)