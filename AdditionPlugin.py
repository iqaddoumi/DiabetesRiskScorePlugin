from typing import List, Dict
from .Base_Calc_Plugin import CalculatorPlugin


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

    def calculate(self, params: Dict[str, any]) -> any:
        num1 = params["key1"]
        num2 = params["key2"]
        return num1 + num2
