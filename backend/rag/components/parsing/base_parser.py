from abc import ABC, abstractmethod
from typing import Generator, Dict, Any

class BaseParser(ABC):

    @abstractmethod
    def parse(self,file_bytes: bytes, filename: str)-> Generator[Dict[str,Any],None,None]:
        """
        Generator[yield return ,sent type,return type]
        
        yield {
        "text": text,
        "metadata": Metadata(BaseModel)
        }
        """
        pass
