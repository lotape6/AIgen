from Indexer import Indexer
from dotenv import load_dotenv
import os
import argparse


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="This script is for indexing some data into a PostgreSQL Vector database", usage="python index_storage.py")

    parser.add_argument('-C', '--clear_database', action='store_true',
                        help="Clear existing database")
    parser.add_argument('-d',
                        '--data_path', default=os.getenv("DATAPATH", "./resources/fragment"), help="Path to data to be indexed")
    parser.add_argument('-R', '--recursive', action='store_true',
                        help="Search in --data_path recursively")
    parser.add_argument(
        '--db_name', default=os.getenv("DBNAME", "default_db"), help="Database name")
    parser.add_argument('--host', default=os.getenv("DBHOST",
                        "localhost"), help="Database host")
    parser.add_argument('--password', default=os.getenv("DBPASSWORD",
                        "password"), help="Database password")
    parser.add_argument(
        '--port', type=int, default=os.getenv("DBPORT", 5432), help="Database port")
    parser.add_argument(
        '--user', default=os.getenv("DBUSER", "user"), help="Database user")
    parser.add_argument(
        '--table_name', default=os.getenv("TABLENAME", "default_table"), help="Table name")
    parser.add_argument('--embedding_model_name', default=os.getenv(
        "EMB_MODEL_NAME", "default_model"), help="Embedding model name")
    parser.add_argument('--vector_size', type=int,
                        default=int(os.getenv("VECTOR_SIZE", 128)), help="Vector size")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Enable verbose mode")

    args = parser.parse_args()

    indexer = Indexer(
        db_name=args.db_name,
        host=args.host,
        password=args.password,
        port=args.port,
        user=args.user,
        table_name=args.table_name,
        embedding_model_name=args.embedding_model_name,
        vector_size=int(args.vector_size),
        verbose=args.verbose,
    )

    if args.clear_database:
        is_deletion_intended = input(f"You are about to delete {args.table_name} table from {args.db_name} database. Are you completely sure!? [y/N]")  # nopep8
        if not (is_deletion_intended == "y" or is_deletion_intended == "Y"):
            exit(0)
        indexer.clearDb()

    indexer.indexDocumentsFromPath(args.data_path, args.recursive)


if __name__ == "__main__":
    main()
