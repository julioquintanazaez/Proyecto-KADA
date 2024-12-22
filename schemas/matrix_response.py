from pydantic import BaseModel
from typing import List


class MatrixResponse(BaseModel):
    _matrix: List[List[int]]
    _list: List[str]


class ListResponse(BaseModel):
    _list: List[str]

