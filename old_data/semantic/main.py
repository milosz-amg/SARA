
import logging
import os
from utils.extractor import EntityExtractor
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def main():
    API_KEY = os.getenv("OPENAI_API_KEY")
    extractor = EntityExtractor(api_key=API_KEY, logger=logger)

    try:
        result = extractor.extract_entities("Find papers by John Doe on machine learning published after 2020.")
        for entity in result.get("entities", []):
            normalized = entity.get("normalized_value")
            if normalized and normalized != entity["value"]:
                print(f"{entity['type']}: {entity['value']} -> {normalized} (confidence: {entity['confidence']})")
            else:
                print(f"{entity['type']}: {entity['value']} (confidence: {entity['confidence']})")
    except Exception as e:
        logger.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()