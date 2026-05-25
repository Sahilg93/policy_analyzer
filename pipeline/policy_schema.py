from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PolicyEvent:
    bill_id: str
    title: str
    state: str
    level: str  # federal / state
    introduced_date: str
    status: str

    text: str

    sponsor: Optional[str] = None
    category: Optional[str] = None

    year: Optional[int] = None