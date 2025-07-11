import pandas as pd
import os
from azure.cosmos.partition_key import PartitionKey
from azure.cosmos import CosmosClient, exceptions
from dotenv import load_dotenv

load_dotenv(override=True)


class AzureCosmos:
    """
    This class has a set of functions defined for uploading and fetching the conversation from the Azure Cosmos DB.
    """

    def __init__(self) -> None:
        """
        Initializes a Azure Cosmos DB object.
        """
        self.COSMOS_HOST = os.getenv(
            "AZURE_COSMOS_DB_URI",
        )
        self.COSMOS_MASTER_KEY = os.getenv("AZURE_COSMOS_DB_KEY")
        self.DATABASE_ID = os.getenv("AZURE_COSMOS_DB_DATABASE_NAME")

        self.client = CosmosClient(
            self.COSMOS_HOST, {"masterKey": self.COSMOS_MASTER_KEY}
        )
        if not self.client:
            print("[ERROR] Cosmos Client not created")
            raise ValueError("Cosmos Client not created")
        self.database = self.client.get_database_client(self.DATABASE_ID)
        self.container = os.getenv("AZURE_COSMOS_DB_CHAT_HISTORY_CONTAINER")

    def initialize_cosmosdb(self, container_name=None) -> None:
        """
        This function is used to initialize the azure cosmos db client and establish the connection to it

        Args:
            container_name (str): The name of the container to be initialized. If None, a ValueError will be raised.

        Returns:
            None
        """
        try:
            if container_name is not None:
                self.container = self.database.get_container_client(container_name)
                if not self.container:
                    print("[ERROR] Cosmos DB container not initialized")
            else:
                print("[ERROR] container_name is None")
                raise ValueError("container_name is None")
        except exceptions.CosmosResourceNotFoundError as e:
            print("[ERROR] CosmosResourceNotFoundError: ", e)
        except Exception as e:
            print("[ERROR] initialize_cosmosdb(): ", e)

    def query_items(self, container_name: str, query: str) -> list:
        """
        This function is used to fetch the conversation based on the SQL Query from the Azure Cosmos DB.

        Args:
            container_name (str): Name of the container on the Cosmos DB database.
            query (str): SQL Query to fetch the content.

        Returns:
            list: List of past 4 conversations.
        """
        try:
            self.initialize_cosmosdb(container_name)
            results = self.container.query_items(
                query=query, enable_cross_partition_query=True
            )
            return results
        except Exception as e:
            print("[ERROR] query_items(): ", e)
            return False

    def upsert_item(self, container_name: str, body: dict) -> bool:
        """
        This function is used to upload the conversation to the Azure Cosmos DB.

        Args:
            container_name (str): Name of the container on Cosmos DB database.
            body (dict): Conversation to be uploaded.

        Returns:
            bool: True if the data is inserted successfully, else False.
        """
        try:
            self.initialize_cosmosdb(container_name)
            response = self.container.upsert_item(body=body)
            if response:
                return True
            return False
        except Exception as e:
            print("[ERROR] upsert_item(): ", e)
            return False

    def create_database(self) -> None:
        """
        This function is used to create the database in azure cosmos db.

        Parameters:
            None

        Returns:
            None
        """
        try:
            self.client.create_database_if_not_exists(self.DATABASE_ID)
            # print("[INFO] database created")
        except Exception as e:
            print("[ERROR] create_database(): ", e)

    def create_container(self, container_name: str, partition_key: str) -> None:
        """
        Creates a container (table) within a database in Azure Cosmos DB if it doesn't exist.

        Args:
            container_name (str): The name of the container to create.
            partition_key (str): The partition key for the container.

        Returns:
            None
        """
        # Create a database if it doesn't exist
        try:
            # self.initialize_cosmosdb()
            self.database.create_container_if_not_exists(
                id=container_name, partition_key=PartitionKey(path=f"/{partition_key}")
            )
            print("[INFO] container created")
        except Exception as e:
            print("[ERROR] Error create_container(): ", e)

    def upload_to_cosmos(self, data_to_insert: dict, container_name: str) -> bool:
        """
        Uploads the conversation data to Azure Cosmos DB.

        Args:
            data_to_insert (dict): The data to insert.
            container_name (str): The name of the container to insert the data to.

        Returns:
            bool: True if the data is inserted successfully, else False.
        """
        try:
            client = CosmosClient(
                self.COSMOS_HOST, {"masterKey": self.COSMOS_MASTER_KEY}
            )
            database = client.get_database_client(database=self.DATABASE_ID)
            container = database.get_container_client(container_name)
            container.upsert_item(data_to_insert)
            print("[INFO] Data uploaded to Cosmos DB successfully")
        except Exception as e:
            print(f"[ERROR] upload_to_cosmos(): {e}")
            return False

    def read_table(self, query: str, container_name: str) -> pd.DataFrame:
        """
        This function is used to read the dataframe from Azure Cosmos DB based on the provided SQL query and container name.

        Args:
            query (str): The SQL query to read the data from the container.
            container_name (str): The name of the container to read the data from.

        Returns:
            pd.DataFrame: The dataframe of the data read from Azure Cosmos DB.
            None: If no results are found.
            False: If an error occurs during the reading process.
        """
        try:
            self.initialize_cosmosdb(container_name=container_name)
            if self.container is None:
                print("[ERROR] Cosmos DB container not initialized")
            items = self.container.query_items(query, enable_cross_partition_query=True)
            results = list(items)
            if not results:
                return None
            return pd.DataFrame(results)
        except Exception as e:
            print("[ERROR] read_table(): ", e)
            return False


if __name__ == "__main__":
    cosmos = AzureCosmos()
    container_name = os.getenv("AZURE_COSMOS_DB_CHAT_HISTORY_CONTAINER")
    print(f"Container Name: {container_name}")
    cosmos.create_container(container_name, "conversation_id")
