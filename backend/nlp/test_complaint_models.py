
from pathlib import Path

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "backend" / "nlp" / "models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_classifier(model_name):
    model_path = MODEL_DIR / model_name

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


def predict_text(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=192,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )[0]

    predicted_id = int(
        torch.argmax(probabilities).item()
    )

    predicted_label = model.config.id2label[
        predicted_id
    ]

    confidence = float(
        probabilities[predicted_id].item()
    )

    return predicted_label, confidence


def main():
    damage_tokenizer, damage_model = load_classifier(
        "damage_classifier"
    )

    urgency_tokenizer, urgency_model = load_classifier(
        "urgency_classifier"
    )

    test_complaints = [
        "Cars are moving into the opposite lane to avoid "
        "a damaged section of road.",

        "A small crack is visible near the edge of the "
        "road, but vehicles are passing normally.",

        "Motorcycles are losing balance when crossing "
        "the damaged road surface.",

        "The road has a network of interconnected cracks "
        "near the market.",

        "A part of the road has sunk and drivers are "
        "slowing down to pass it.",

        "The road surface is uneven near the entrance "
        "of the residential area.",

        "There is a broken drainage cover on the road "
        "near the hospital.",

        "A wide crack crosses the road and causes "
        "vehicles to brake suddenly.",
    ]

    results = []

    print("\nRoadSight AI NLP Model Testing")
    print("=" * 70)
    print("Device:", DEVICE)

    for text in test_complaints:
        damage, damage_confidence = predict_text(
            text,
            damage_tokenizer,
            damage_model,
        )

        urgency, urgency_confidence = predict_text(
            text,
            urgency_tokenizer,
            urgency_model,
        )

        print("\nComplaint:")
        print(text)

        print(
            f"Damage type: {damage} "
            f"({damage_confidence:.1%})"
        )

        print(
            f"Urgency: {urgency} "
            f"({urgency_confidence:.1%})"
        )

        results.append({
            "text": text,
            "damage_type": damage,
            "damage_confidence": damage_confidence,
            "urgency": urgency,
            "urgency_confidence": urgency_confidence,
        })

    output_file = (
        MODEL_DIR / "manual_test_predictions.csv"
    )

    pd.DataFrame(results).to_csv(
        output_file,
        index=False,
    )

    print("\nTesting completed.")
    print("Results saved to:")
    print(output_file)


if __name__ == "__main__":
    main()
