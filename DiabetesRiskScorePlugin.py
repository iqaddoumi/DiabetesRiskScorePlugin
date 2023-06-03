from typing import List, Dict
import pandas as pd
from .Base_Calc_Plugin import CalculatorPlugin
from .Setup_Model import load_model, convert_to_features


class AdditionPlugin(CalculatorPlugin):
    @classmethod
    def get_name(cls) -> str:
        return "Calculated Sum"

    @classmethod
    def required_parameters(cls) -> List[str]:
        return ["key1", "key2"]

    @classmethod
    def get_entity_type(cls) -> EntityType:
        return EntityType.NUMERICAL

    def calculate(df, disease="diabetes"):
        # Values weren't passed, load from disk
        model, encoder, scaler = load_model(disease, encoder=disease != 'CHD')

        if isinstance(df, dict):
            df = pd.DataFrame(df, index=[0])
            has_disease, risk_score = df.calculate(df, disease)
            return has_disease[0], risk_score[0]

        if disease != "CHD":
            df = convert_to_features(df, encoder)
        # Scale the input features using the same scaler as used in training
        new_patient_x_scaled = scaler.transform(df)

        risk_score = model.predict_proba(new_patient_x_scaled)

        print('Risk score:', risk_score[:, 1])

        has_disease = risk_score[:, 1] >= 0.5

        return has_disease, risk_score[:, 1]



