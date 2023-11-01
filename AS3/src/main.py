import os
import sys

file_location = os.path.abspath(__file__)  # Get current file abspath
root_directory = os.path.dirname(file_location)  # Get root dir
sys.path.append(os.path.join(root_directory, '..'))
from tools.scraping_data.scraping_bitcoin_RNN import *


def main():
    start = Date(2022, 3, 1, 0)
    end = Date(2023, 11, 14, 0)
    start.print()
    print(start.get_timestamp())
    end.print()
    print(end.get_timestamp())
    # Print request numbers
    print("\n%d request number\n" % len([k for k in range(start.get_timestamp(), end.get_timestamp(), INTERVAL_MS)]))
    df = get_data_klines(start, end)
    export_xlsx(os.path.join(root_directory, '..', '..', 'dataset', 'BTCUSDT.xlsx'), df)


if __name__ == "__main__":
    main()
