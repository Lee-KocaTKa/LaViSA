from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types 
from tqdm import tqdm

categories = ["vp", "anaph", "ellip", "conj", "adjscope", "verbscope"] 
categories = ["pp"] 
def load_dataset(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept either:
    # 1. [ {...}, {...} ]
    # 2. { "data": [ {...}, {...} ] }
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]

    raise ValueError("Unsupported JSON format. Expected list or dict with key 'data'.")

def open_image_rgb(image_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        
    return types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png",
    )

def encode_one(
    client,
    item
) -> np.ndarray:
    
   

    result = client.models.embed_content(
        model="gemini-embedding-2", 
        contents=[item]
    )
    emb = result.embeddings[0].values
    
    return np.array(emb, dtype=np.float32) 

def main() -> None:
    
    
    
    
    for category in categories: 
        json_path = "../../../data/ViLStrUB/jsons_UNIT2/"
        image_dir = "../../../data/ViLStrUB/images/"
        outputpath = "../../../data/ViLStrUB/geminiembeddings/"
        output: dict[str, Any] = {
            "metadata": {
                "category": category,
                "model_name": "gemini-embedding-2",
                "json_path": str(json_path),
                "image_dir": str(image_dir),
                "normalized": True,
                "embedding_dtype": "float32",
                "structure": {
                "ambiguous_text": "embedding of original ambiguous sentence",
                "clarified_text": "embedding of each resolved / rewritten sentence",
                "image": "embedding of each image only",
                "ambiguous_plus_image": "embedding of ambiguous sentence + image",
            },
        },
        "groups": {},
    }
        print(f"Processing category: {category}")
        json_path = Path(json_path) / f"{category}.json"
        image_dir = Path(image_dir) / category

        output_path = Path(outputpath) / f"gemini_embeddings_{category}.pkl"
        dataset = load_dataset(json_path)
        key = input("Enter your Gemini API key: ").strip()
        client = genai.Client(api_key=key) 

        

        for group in tqdm(dataset, desc="Encoding groups"):
            group_id = group["GroupID"]
            ambiguous_sentence = group["Sentence"]

            group_record: dict[str, Any] = {
                "ambig_sentence": ambiguous_sentence,
                "style": group.get("Style"),
                "embedding": encode_one(
                    client,
                    ambiguous_sentence
            ),
            "variants": {},
        }

            for variant in group["Variants"]:
                sentence_id = variant["SentenceID"]

            
                clarified_sentence = variant["Meaning"]

                image_path = image_dir / variant["Image"]
                image = open_image_rgb(image_path)

                variant_record = {
                    "sentence_id": sentence_id,
                    "meaning": variant.get("Meaning"),
                    "image_file": variant["Image"],

                # 2. only clarified sentence
                    "clarified_embedding": encode_one(
                        client,
                        clarified_sentence,
                    ),

                # 3. only image
                    "image_embedding": encode_one(
                        client,
                        image,
                    ),

                # 4. ambiguous sentence + each image
                    "ambiguous_plus_image_embedding": encode_one(
                        client,
                        {"text": ambiguous_sentence, "image": image},
                    ),
                }

                group_record["variants"][sentence_id] = variant_record

            output["groups"][group_id] = group_record

    #args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pickle preserves numpy arrays directly.
        with output_path.open("wb") as f:
            pickle.dump(output, f)

        print(f"Saved embeddings to: {output_path}")


if __name__ == "__main__":
    main()