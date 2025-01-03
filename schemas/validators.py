from pydantic import BaseModel
from datetime import datetime

# Create your schemas here.

class DateTimeModel(BaseModel):
    fecha: datetime