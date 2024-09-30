from typing import Optional
from pydantic_settings import BaseSettings
import os
import requests

class ClassSettings(BaseSettings):
    DATABASE_URL: Optional[str] = None

    def set_database_url(self, value: Optional[str] = None) -> str:
        if not value:
            #ToDO: Add logger
            #loger.debug("No Database URL is provided, checking the env value")
            print("No Database URL is provided, checking the env value")
            if database_url := os.getenv("CSA_DATABASE_URL"):
                self.DATABASE_URL = database_url
                db_type = os.getenv("DB_TYPE")
                #loger.info(f"Database type is {db_type} URL set to {database_url}")
                print(f"Database type is {db_type} URL set to database_url")
                return self.DATABASE_URL
            else:
                #logger.debug("No DATABASE_URL env variable, using sqlite database")
                print("No DATABASE_URL env variable, using sqlite database")
                if os.path.exists("Chinook.db"):
                     print("File already exists")
                else:
                    print("Downloading the sample Chinook db file")
                    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

                    response = requests.get(url)

                    if response.status_code == 200:
                    # Open a local file in binary write mode
                        with open("Chinook.db", "wb") as file:
                            # Write the content of the response (the file) to the local file
                            file.write(response.content)
                        print("File downloaded and saved as Chinook.db")
                    else:
                        print(f"Failed to download the file. Status code: {response.status_code}")
                    self.DATABASE_URL = "sqlite:///Chinook.db"
                    return self.DATABASE_URL
        else:
            self.DATABASE_URL = value
        return self.DATABASE_URL
