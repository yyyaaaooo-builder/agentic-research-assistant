from abc import ABC, abstractmethod


class Tool(ABC):
    name: str
    description: str
    parameters: dict

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool and return its result as a string.

        Implementations should report failures (bad input, etc.) as error
        text in the returned string rather than raising.
        """
