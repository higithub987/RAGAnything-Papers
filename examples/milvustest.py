from pymilvus import connections, utility


def test_milvus_connection():
    try:
        # Connect to the Milvus container running on localhost
        # Port 19530 is the default gRPC port for Milvus
        connections.connect(alias="default", host="localhost", port="19530")

        # Check connection and get server version
        if utility.has_collection("non_existent_collection"):
            pass  # Just a dummy check to ensure communication works

        version = utility.get_server_version()
        print("✅ Connected successfully to Milvus!")
        print(f"🚀 Server Version: {version}")

    except Exception as e:
        print("❌ Failed to connect to Milvus.")
        print(f"Error details: {e}")


if __name__ == "__main__":
    test_milvus_connection()
