from src.pull_fred_data import load_fred_data
from src.transformations import transform_data


def main():
    load_fred_data()
    transform_data()


if __name__ == "__main__":
    main()
