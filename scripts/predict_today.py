from stock_ai.api.services.predict_service import get_ranking


def main():

    ranking = get_ranking()

    print("\n=== Today's Ranking ===\n")

    print(
        ranking.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()