from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

#URL_DATABASE='mysql+pymysql://root:Pak@123@localhost:3306/neta_ai'
#URL_DATABASE = 'mysql+pymysql://root:Umar%40123@3.89.28.11:3306/neta_ai'


URL_DATABASE = 'mysql+pymysql://root:@127.0.0.1:3306/netapython'

#DATABASE_URL = "mysql+pymysql://root:123Secure%21@localhost/neta_ai"
engine=create_engine(URL_DATABASE)
SessionLocal=sessionmaker(autocommit=False,autoflush=False,bind=engine)
Base=declarative_base()



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
