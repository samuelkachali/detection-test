import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load OpenAI's pre-trained CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. Load your image
image_path = r"C:\Users\USER\Documents\Trusam\testing\Soybean-Septoria-brown-spot-1-Craig-Grau-and-University-of-Wisconsin-Teaching-Images_1280x720_acf_cropped.jpg"
image = Image.open(image_path).convert("RGB")

# 3. Define explicit text prompts for your specific classes
# --- REVISED CLIP PROMPTS WITH ACTUAL TARGET ---
candidate_labels = [
    "a clean healthy green soybean leaf without any spots",
    "a soybean leaf infected with septoria brown spot showing tiny dark brown flecks",
    "a soybean leaf with bacterial pustule lesions",
    "a maize leaf showing brown common rust pustules",
    "a bean leaf with angular leaf spot geometric lesions"
]

# 4. Run inference
inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    
# 5. Get probability scores using softmax
logits_per_image = outputs.logits_per_image 
probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

# 6. Print the results
print("--- CLIP Zero-Shot Results ---")
for label, score in zip(candidate_labels, probs):
    print(f"{label}: {score:.1%}")