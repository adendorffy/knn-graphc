from pathlib import Path
import argparse
import csv
import zipfile


def load_scores(score_path: Path) -> dict[str, float]:
    scores = {}

    with score_path.open() as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            filename, score = line.split()
            scores[filename] = float(score)

    return scores


def load_gold_from_zip(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("lexical/dev/gold.csv") as f:
            # Text wrapper because ZipFile gives us bytes
            import io

            reader = csv.DictReader(
                io.TextIOWrapper(f, encoding="utf-8")
            )

            return list(reader)


def score_lexical(
    zip_path: Path,
    score_path: Path,
):
    scores = load_scores(score_path)
    gold = load_gold_from_zip(zip_path)

    # ---------------------------------------------------------
    # Organise by (item id, voice)
    #
    # Each pair should contain:
    #   correct=1 : real word
    #   correct=0 : corresponding nonword
    # ---------------------------------------------------------
    pairs = {}

    for row in gold:
        key = (
            row["id"],
            row["voice"],
        )

        correct = int(row["correct"])
        filename = row["filename"]

        if key not in pairs:
            pairs[key] = {}

        pairs[key][correct] = {
            "filename": filename,
            "word": row["word"],
            "frequency": row["frequency"],
            "length": row["length"],
        }

    n_correct = 0
    n_pairs = 0
    n_missing = 0
    n_ties = 0

    results = []

    for (item_id, voice), pair in pairs.items():

        if 1 not in pair or 0 not in pair:
            print(
                f"WARNING: incomplete pair "
                f"id={item_id}, voice={voice}"
            )
            continue

        word = pair[1]
        nonword = pair[0]

        word_key = word["filename"]
        nonword_key = nonword["filename"]

        if (
            word_key not in scores
            or nonword_key not in scores
        ):
            n_missing += 1
            continue

        word_score = scores[word_key]
        nonword_score = scores[nonword_key]

        if word_score > nonword_score:
            correct = 1
            n_correct += 1

        elif word_score < nonword_score:
            correct = 0

        else:
            # Exact tie.
            correct = 0.5
            n_correct += 0.5
            n_ties += 1

        n_pairs += 1

        results.append(
            {
                "id": item_id,
                "voice": voice,
                "word": word["word"],
                "nonword": nonword["word"],
                "frequency": word["frequency"],
                "length": word["length"],
                "word_score": word_score,
                "nonword_score": nonword_score,
                "correct": correct,
            }
        )

    if n_pairs == 0:
        raise RuntimeError(
            "No complete scored lexical pairs found."
        )

    accuracy = n_correct / n_pairs

    print()
    print("=" * 60)
    print("SLM21 LEXICAL DEV")
    print("=" * 60)

    print(f"Scores loaded:       {len(scores):,}")
    print(f"Gold rows:           {len(gold):,}")
    print(f"Scored pairs:        {n_pairs:,}")
    print(f"Missing pairs:       {n_missing:,}")
    print(f"Ties:                {n_ties:,}")
    print(f"Correct:             {n_correct:,.1f}")
    print()
    print(
        f"Lexical accuracy:    "
        f"{accuracy:.4f} "
        f"({100 * accuracy:.2f}%)"
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    score_lexical(
        zip_path=args.zip_path,
        score_path=args.scores,
    )