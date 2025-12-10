from igel.medical.models.pneumonia_model import PneumoniaClassifier
import sys

if __name__ == "__main__":
    image_path = sys.argv[1]  # pass image path as CLI arg
    model = PneumoniaClassifier()
    result = model.predict(image_path)
    print(f"Prediction: {result}")
