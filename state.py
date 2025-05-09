from typing import TypedDict, List, Optional, Annotated

class Food(TypedDict):
    favourites: Optional[List[str]]
    dislikes: Optional[List[str]]

class Diseases(TypedDict):
    NCDs: Optional[List[str]]
    allergies: Optional[List[str]]

class Profile(TypedDict):
    name: str
    age: int
    country: str
    diseases: Diseases
    food: Food

from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str