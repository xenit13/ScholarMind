from scholar_mind.config.settings import get_settings
from scholar_mind.db.session import init_database


def main() -> None:
    init_database(get_settings())


if __name__ == "__main__":
    main()
