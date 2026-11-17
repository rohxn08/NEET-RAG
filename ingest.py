"""Entry point: python ingest.py [--data-dir data]"""
import argparse
from src.ingest import run_ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Folder containing PDFs and CSVs")
    args = parser.parse_args()
    run_ingest(args.data_dir)
