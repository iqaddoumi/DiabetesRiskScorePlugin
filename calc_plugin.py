from abc import abstractmethod, ABC
from typing import Dict, List


class CalculatorPlugin(ABC):
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def required_parameters(cls) -> List[str]:
        pass

    @classmethod
    @abstractmethod
    def get_entity_type(cls) -> EntityType:
        pass

    @abstractmethod
    def calculate(self, params: Dict[str, any]) -> any:
        pass


