from sqlalchemy import Boolean, Column, ForeignKey, Numeric, Integer, String
from sqlalchemy.orm import relationship

from database import Base

class Client(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(String)
    confidenceLevel = Column(Numeric(10, 2))
    confidenceLevel1 = Column(Numeric(10, 2))
    confidenceLevel2 = Column(Numeric(10, 2))
    confidenceLevel3 = Column(Numeric(10, 2))
    confidenceLevel4 = Column(Numeric(10, 2))
    confidenceLevel5 = Column(Numeric(10, 2))
    confidenceLevel6 = Column(Numeric(10, 2))
    confidenceLevel7 = Column(Numeric(10, 2))
    confidenceLevel8 = Column(Numeric(10, 2))