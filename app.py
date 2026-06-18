import os
import gc
# Optimizations for local CPU execution environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import tensorflow as tf
import torch
torch.set_num_threads(1)
if hasattr(tf, 'config') and hasattr(tf.config, 'experimental'):
    # Prevent TensorFlow from allocating all memory upfront
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import json
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO  # NEW: YOLO Object Detection Core

# --- PAGE CONFIG ---
st.set_page_config(page_title="Model Comparison Pipeline - Crop AI", page_icon="🌱", layout="wide")

# --- CUSTOM CNN MODEL DEFINITION (Matches saved PyTorch weights) ---

class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


#  CLIP ENSEMBLE PROMPT MAP 

CLIP_ENSEMBLE_MAP = {
    'Tomato___Late_blight': [
        "a tomato leaf with dark water-soaked spots from late blight disease",
        "late blight fungal infection destroying a tomato plant leaf",
        "a decaying tomato leaf affected by phytophthora late blight"
    ],
    'Tomato___healthy': [
        "a clean healthy green tomato leaf",
        "a pristine green tomato leaf without any disease or blemishes",
        "healthy tomato foliage"
    ],
    'Grape___healthy': [
        "a clean healthy green grape leaf",
        "healthy vibrant grape vine foliage",
        "a pristine grape leaf without any fungal spots"
    ],
    'Orange___Haunglongbing_(Citrus_greening)': [
        "a citrus orange leaf showing yellow mottling from huanglongbing or greening disease",
        "citrus greening disease causing asymmetric yellowing on an orange leaf",
        "an orange tree leaf infected with HLB citrus greening"
    ],
    'Soybean___healthy': [
        "a clean healthy green soybean leaf with zero spots",
        "pristine green soy plant foliage",
        "a perfectly healthy soybean leaf surface"
    ],
    'Squash___Powdery_mildew': [
        "a squash leaf covered in white powdery mildew fungus",
        "powdery mildew creating white dusty patches on a squash leaf",
        "a pumpkin or squash leaf showing white fungal spots"
    ],
    'Potato___healthy': [
        "a healthy green potato plant leaf",
        "pristine potato leaf foliage",
        "a clean potato leaf free of blight spots"
    ],
    'Corn_(maize)___Northern_Leaf_Blight': [
        "a corn leaf with long cigar-shaped northern leaf blight lesions",
        "maize foliage showing elongated grayish-brown northern leaf blight stripes",
        "northern corn leaf blight fungal damage on a maize leaf"
    ],
    'Tomato___Early_blight': [
        "a tomato leaf with dark concentric rings from early blight disease",
        "early blight target-like brown spots on a tomato leaf",
        "alternaria early blight infection yellowing a tomato leaf"
    ],
    'Tomato___Septoria_leaf_spot': [
        "a tomato leaf with numerous tiny circular spots showing gray centers",
        "septoria leaf spot fungal infection covering a tomato leaf",
        "small dark concentric flecks of septoria on tomato foliage"
    ],
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
        "a corn leaf with rectangular gray leaf spot lesions running parallel to veins",
        "maize leaf infected with cercospora gray leaf spot disease",
        "grayish rectangular blocky spots on a corn leaf"
    ],
    'Strawberry___Leaf_scorch': [
        "a strawberry leaf with purplish-brown blotches from leaf scorch",
        "leaf scorch disease drying out a strawberry leaf",
        "purplish lesions spreading across a strawberry leaf surface"
    ],
    'Peach___healthy': [
        "a clean healthy elongated peach tree leaf",
        "healthy pristine peach leaf foliage",
        "a peach leaf free of bacterial spots"
    ],
    'Apple___Apple_scab': [
        "an apple leaf with velvety olive-green to brown scab spots",
        "apple scab fungal lesions roughing up an apple leaf",
        "venturia inaequalis scab infection on an apple tree leaf"
    ],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        "a tomato leaf showing severe curling wrinkling and yellow margins",
        "tomato yellow leaf curl virus causing stunted deformed leaves",
        "curled upward deformed yellowing tomato foliage from virus"
    ],
    'Tomato___Bacterial_spot': [
        "a tomato leaf covered in small greasy water-soaked bacterial spots",
        "bacterial spot disease creating dark lesions with yellow halos on a tomato leaf",
        "xanthomonas bacterial spot damage on tomato foliage"
    ],
    'Apple___Black_rot': [
        "an apple leaf showing frog-eye circular brown spots from black rot",
        "black rot fungal infection causing concentric rings on an apple leaf",
        "frog-eye leaf spot black rot on apple foliage"
    ],
    'Blueberry___healthy': [
        "a healthy smooth blueberry plant leaf",
        "clean green blueberry foliage",
        "pristine blueberry leaf without spots"
    ],
    'Cherry_(including_sour)___Powdery_mildew': [
        "a cherry tree leaf with white powdery mildew coating",
        "powdery mildew fungus distorting a cherry leaf",
        "dusty white fungal patches on cherry foliage"
    ],
    'Peach___Bacterial_spot': [
        "a peach tree leaf with small dark spots that leave shot-holes",
        "bacterial spot causing deep dark lesions on an elongated peach leaf",
        "peach foliage infected with xanthomonas bacterial spot"
    ],
    'Apple___Cedar_apple_rust': [
        "an apple leaf with bright orange-yellow circular rust spots",
        "cedar apple rust disease producing striking orange spots on an apple leaf",
        "bright yellow-orange fungal lesions on apple foliage"
    ],
    'Tomato___Target_Spot': [
        "a tomato leaf showing small dark brown spots with subtle target rings",
        "target spot corynespora disease on a tomato plant leaf",
        "brown circular spots with faint concentric circles on tomato foliage"
    ],
    'Pepper,_bell___healthy': [
        "a healthy shiny green bell pepper leaf",
        "clean pristine capsicum bell pepper foliage",
        "a healthy green pepper plant leaf"
    ],
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': [
        "a grape leaf with large dark brown dead patches from leaf blight",
        "isariopsis leaf blight fungal damage dry spots on a grape leaf",
        "large expanding brown lesions on grape vine foliage"
    ],
    'Potato___Late_blight': [
        "a potato leaf with dark water-soaked spots from late blight disease",
        "late blight phytophthora infection blackening a potato leaf",
        "decaying dying potato foliage from late blight attack"
    ],
    'Tomato___Tomato_mosaic_virus': [
        "a tomato leaf showing dark and light green mottled mosaic patterns",
        "tomato mosaic virus causing blistered mottled green patterns on a leaf",
        "mottled green and yellow mosaic virus markings on tomato foliage"
    ],
    'Strawberry___healthy': [
        "a healthy green three-lobed strawberry leaf",
        "clean green strawberry plant foliage",
        "pristine strawberry leaf surface without scorch"
    ],
    'Apple___healthy': [
        "a clean healthy green apple tree leaf",
        "pristine green apple leaf foliage",
        "healthy orchard apple tree leaf"
    ],
    'Grape___Black_rot': [
        "a grape leaf with small reddish-brown circular spots with dark borders",
        "black rot fungal lesions attacking a grape vine leaf",
        "small brown necrotic spots on a grape leaf surface"
    ],
    'Potato___Early_blight': [
        "a potato leaf with target-board dark brown concentric circles",
        "early blight alternaria spots on a lower potato plant leaf",
        "brown circular target-like lesions on potato foliage"
    ],
    'Cherry_(including_sour)___healthy': [
        "a clean healthy green cherry tree leaf",
        "pristine cherry plant foliage",
        "healthy sour cherry tree leaf surface"
    ],
    'Corn_(maize)___Common_rust_': [
        "a corn leaf covered in powdery golden-brown cinnamon rust blisters",
        "maize leaf showing raised reddish-brown common rust pustules",
        "corn rust spores dusting a maize crop leaf"
    ],
    'Grape___Esca_(Black_Measles)': [
        "a grape leaf showing tiger-stripe yellow and brown drying patterns",
        "esca black measles disease causing dramatic leaf tissue necrosis on grape foliage",
        "tiger-striped drying wilting grape vine leaf"
    ],
    'Raspberry___healthy': [
        "a healthy green textured raspberry leaf",
        "clean pristine raspberry plant foliage",
        "healthy raspberry leaf surface free of disease"
    ],
    'Tomato___Leaf_Mold': [
        "a tomato leaf with pale green or yellow spots showing olive-green mold underneath",
        "tomato leaf mold fungal coating fading a leaf surface",
        "velvety mold growth causing pale patches on tomato foliage"
    ],
    'Tomato___Spider_mites Two-spotted_spider_mite': [
        "a tomato leaf showing tiny yellow stippling speckles and fine webbing",
        "two-spotted spider mite damage bleaching or speckling a tomato leaf",
        "stippled yellow dried out tomato leaf from spider mite feeding"
    ],
    'Pepper,_bell___Bacterial_spot': [
        "a bell pepper leaf with small irregular dark spots or cracks",
        "bacterial spot causing yellow-green pimple-like spots on a pepper leaf",
        "capsicum bell pepper foliage showing xanthomonas bacterial spot"
    ],
    'Corn_(maize)___healthy': [
        "a clean healthy green corn leaf",
        "pristine long green maize plant leaf",
        "healthy corn foliage without any stripes or rust blisters"
    ],
    'maize_common_rust': [
        "a corn leaf covered in powdery golden-brown cinnamon rust blisters",
        "maize leaf showing raised reddish-brown common rust pustules",
        "corn rust spores dusting a maize crop leaf"
    ],
    'maize_healthy': [
        "a clean healthy green corn leaf",
        "pristine long green maize plant leaf",
        "healthy corn foliage without any stripes or rust blisters"
    ],
    'maize_blight': [
        "a corn leaf with long cigar-shaped northern leaf blight lesions",
        "maize foliage showing elongated grayish-brown northern leaf blight stripes",
        "northern corn leaf blight fungal damage on a maize leaf"
    ],
    'maize_gray_leaf_spot': [
        "a corn leaf with rectangular gray leaf spot lesions running parallel to veins",
        "maize leaf infected with cercospora gray leaf spot disease",
        "grayish rectangular blocky spots on a corn leaf"
    ],
    'beans_rust': [
        "a photo of common bean rust, absolutely not a soybean crop",
        "common green beans showing rust, distinct from soyabeans",
        "a common green bean leaf with raised reddish-brown rust spore pustules",
        "phaseolus vulgaris bean plant showing rust spots next to smooth green bean pods",
        "garden bean foliage with powdery golden rust blisters running along leaf veins",
        "dry edible bean crop showing brown rust fungal breakout on leaves",
        "smooth pods", "characteristic leaf texture", "non-pubescent stems",
        "Brown, black or white spore pustules on leaves, stems or pods"
    ],
    'beans_healthy': [
        "a healthy green common bean leaf without any spots",
        "pristine smooth green bean plant foliage and healthy pods",
        "clean phaseolus vulgaris bean leaves free of disease"
    ],
    'soyabeans_rust': [
        "a soybean leaf with small red-brown volcanic rust pustules on the underside",
        "asian soybean rust disease causing severe yellowing on fuzzy soy plant foliage",
        "brown powdery rust blisters covering a soybean leaf near hairy soybean pods",
        "glycine max agricultural field crop infected with soybean rust",
        "hairy structures", "fuzzy pods", "volcanic rust pustules"
    ],
    'soyabeans_healthy': [
        "a clean healthy green soybean leaf with zero spots",
        "pristine fuzzy green soy plant foliage",
        "a perfectly healthy glycine max soybean leaf surface"
    ],
    'soyabeans_suddendeathsyndrome': [
        "a soybean leaf showing severe yellow blotches between veins with a green midrib",
        "sudden death syndrome causing bright interveinal necrosis on a soybean leaf",
        "soybean leaf dying off with bright yellow-brown stripes between green veins"
    ],
    'soyabeans_frogeyeleafspot': [
        "a soybean leaf with small circular gray spots ringed by dark reddish-brown borders",
        "frog-eye leaf spot fungal lesions on a soybean plant leaf",
        "circular tan to gray spots resembling a frog eye on soy foliage"
    ],
    'soyabeans_yellow_mosaic': [
        "a soybean leaf with bright yellow patches and green mosaic mottling",
        "soybean yellow mosaic virus bright yellowing spreading on a leaf",
        "mottled bright yellow and green distorted soybean plant foliage"
    ],
    'soyabeans_targetleafspot': [
        "a soybean leaf with large brown circular lesions showing prominent concentric rings",
        "target leaf spot corynespora disease on a soybean leaf",
        "expanding dark brown circular spots resembling a target on a soy leaf"
    ],
    'soyabeans_bacterialpustule': [
        "a soybean leaf with small light green spots showing raised bumpy centers",
        "bacterial pustule infection creating tiny elevated bumps on a soybean leaf",
        "soybean foliage displaying yellow specks with elevated volcanic centers"
    ],
    'unknown_or_other': [
        "a random object that is not a plant leaf",
        "a picture of a person, face, room, or everyday object",
        "human clothing, electronics, buildings, or text graphics",
        "blurry background noise or scenery with no crops present"
    ]
}

#  CLEAN DISPLAY NAMES FOR THE UI 
DISPLAY_NAME_MAP = {
    'Tomato___Late_blight': 'Tomato - Late Blight',
    'Tomato___healthy': 'Tomato - Healthy',
    'Grape___healthy': 'Grape - Healthy',
    'Orange___Haunglongbing_(Citrus_greening)': 'Citrus - Citrus Greening (HLB)',
    'Soybean___healthy': 'Soybean - Healthy',
    'Squash___Powdery_mildew': 'Squash - Powdery Mildew',
    'Potato___healthy': 'Potato - Healthy',
    'Corn_(maize)___Northern_Leaf_Blight': 'Maize - Northern Leaf Blight',
    'Tomato___Early_blight': 'Tomato - Early Blight',
    'Tomato___Septoria_leaf_spot': 'Tomato - Septoria Leaf Spot',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Maize - Gray Leaf Spot',
    'Strawberry___Leaf_scorch': 'Strawberry - Leaf Scorch',
    'Peach___healthy': 'Peach - Healthy',
    'Apple___Apple_scab': 'Apple - Apple Scab',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato - Yellow Leaf Curl Virus',
    'Tomato___Bacterial_spot': 'Tomato - Bacterial Spot',
    'Apple___Black_rot': 'Apple - Black Rot',
    'Blueberry___healthy': 'Blueberry - Healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'Cherry - Powdery Mildew',
    'Peach___Bacterial_spot': 'Peach - Bacterial Spot',
    'Apple___Cedar_apple_rust': 'Apple - Cedar Apple Rust',
    'Tomato___Target_Spot': 'Tomato - Target Spot',
    'Pepper,_bell___healthy': 'Bell Pepper - Healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape - Leaf Blight',
    'Potato___Late_blight': 'Potato - Late Blight',
    'Tomato___Tomato_mosaic_virus': 'Tomato - Mosaic Virus',
    'Strawberry___healthy': 'Strawberry - Healthy',
    'Apple___healthy': 'Apple - Healthy',
    'Grape___Black_rot': 'Grape - Black Rot',
    'Potato___Early_blight': 'Potato - Early Blight',
    'Cherry_(including_sour)___healthy': 'Cherry - Healthy',
    'Corn_(maize)___Common_rust_': 'Maize - Common Rust',
    'Grape___Esca_(Black_Measles)': 'Grape - Esca (Black Measles)',
    'Raspberry___healthy': 'Raspberry - Healthy',
    'Tomato___Leaf_Mold': 'Tomato - Leaf Mold',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato - Spider Mites',
    'Pepper,_bell___Bacterial_spot': 'Bell Pepper - Bacterial Spot',
    'Corn_(maize)___healthy': 'Maize - Healthy',
    'unknown_or_other': 'Unknown / Not a Plant Leaf',
    'maize_common_rust': 'Maize - Common Rust',
    'maize_healthy': 'Maize - Healthy',
    'maize_blight': 'Maize - Leaf Blight',
    'maize_gray_leaf_spot': 'Maize - Gray Leaf Spot',
    'beans_rust': 'Beans - Rust',
    'beans_angular_leaf_spot': 'Beans - Angular Leaf Spot',
    'beans_healthy': 'Beans - Healthy',
    'soyabeans_rust': 'Soybeans - Rust',
    'soyabeans_healthy': 'Soybeans - Healthy',
    'soyabeans_suddendeathsyndrome': 'Soybeans - Sudden Death Syndrome',
    'soyabeans_frogeyeleafspot': 'Soybeans - Frogeye Leaf Spot',
    'soyabeans_yellow_mosaic': 'Soybeans - Yellow Mosaic',
    'soyabeans_targetleafspot': 'Soybeans - Target Leaf Spot',
    'soyabeans_bacterialpustule': 'Soybeans - Bacterial Pustule'
}

def format_disease_name(raw_name):
    """Fallback function that formats the name if it's missing from the map"""
    if raw_name in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[raw_name]
    cleaned = raw_name.replace("___", " - ").replace("_", " ")
    return cleaned.title()


# ASSET LOADING ENGINE 

@st.cache_resource 
def load_all_assets():
    # 1. Load V1 (TensorFlow Keras Multi-Head)
    v1_model = tf.keras.models.load_model("crop_disease_v4.keras")
    
    # 2. Load V2 (TensorFlow Keras Single-Head)
    v2_model = tf.keras.models.load_model("crop_disease_unified_v2.keras")
    with open('class_indices.json', 'r') as f:
        class_map = json.load(f)
    v2_labels = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]
    
    # 3. Load V3 (PyTorch Custom CNN)
    with open('pytoch_indices.json', 'r') as f:
        pt_idx_to_class = json.load(f)
    num_classes = len(pt_idx_to_class)
    
    pt_model = CNN_NeuralNet(3, num_classes)
    pt_model.load_state_dict(torch.load("crop_disease_CNN_v3.pth", map_location=torch.device('cpu')))
    pt_model.eval()
    
    # 4. Load OpenAI CLIP Foundation Pipeline
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 5. NEW: Load YOLO Localization Model (Can switch weights file paths as needed)
    base_path = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    yolo_weight_path = os.path.join(base_path, "best.pt")

    if not os.path.exists(yolo_weight_path):
        st.error(f"🚨 Critical Asset Missing: Put your weights file at {yolo_weight_path}")
        st.stop()

    # Load your custom trained model weights
    yolo_model = YOLO(yolo_weight_path)
    
    return v1_model, v2_model, v2_labels, pt_model, pt_idx_to_class, clip_model, clip_processor, yolo_model

# Extract running assets
v1_model, v2_model, v2_labels, pt_model, pt_classes, clip_model, clip_processor, yolo_model = load_all_assets()

#  PROMPT ENSEMBLE MATH INFERENCE ENGINE
def run_clip_ensemble_inference(image, target_class_list):
    flat_prompts = []
    prompt_to_class_idx = []
    
    for idx, class_name in enumerate(target_class_list):
        sentences = CLIP_ENSEMBLE_MAP.get(class_name, [f"a photo of a leaf with {class_name}"])
        for sentence in sentences:
            flat_prompts.append(sentence)
            prompt_to_class_idx.append(idx)
            
    inputs = clip_processor(text=flat_prompts, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image 
        flat_probabilities = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
    class_accumulated_scores = np.zeros(len(target_class_list))
    class_sentence_counts = np.zeros(len(target_class_list))
    
    for prob, class_idx in zip(flat_probabilities, prompt_to_class_idx):
        class_accumulated_scores[class_idx] += prob
        class_sentence_counts[class_idx] += 1
        
    final_class_probabilities = class_accumulated_scores / np.maximum(class_sentence_counts, 1)
    final_class_probabilities = final_class_probabilities / np.sum(final_class_probabilities)
    
    best_idx = np.argmax(final_class_probabilities)
    return target_class_list[best_idx], final_class_probabilities[best_idx]


#  PREPROCESSING & APP INTERFACE 

pytorch_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

V1_CROP_NAMES = ['bean', 'maize', 'soya']
V1_DISEASE_NAMES = [
    'Maize Gray Leaf Spot', 'Soybeans Healthy', 'Soybeans Bacterial Pustule', 
    'Soybeans Frog Eye Leaf Spot', 'Maize Blight', 'Soybeans Sudden Death Syndrome', 
    'Beans Angular Leaf Spot', 'Soybeans Rust', 'Beans Rust', 
    'Soybeans Target Leaf Spot', 'Maize Common Rust', 'Maize Healthy', 
    'Beans Healthy', 'Soybeans Yellow Mosaic'
]

st.title("🌱 AI Crop Disease Detection: Pipeline Comparison Engine")
st.write("Cross-framework analysis: Standard CNN Classifier Layers vs Semantic Vision Foundations vs Object Detection.")

uploaded_file = st.sidebar.file_uploader("Upload leaf image for system evaluation", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open base target image non-destructively
    image = Image.open(uploaded_file).convert('RGB')
    st.sidebar.image(image, caption='Original Input Image', use_container_width=True)
    
    # Formulate Base Tensors / Arrays for classic workflows
    img_tf = image.resize((224, 224))
    img_array_tf = tf.keras.utils.img_to_array(img_tf) / 255.0
    img_array_tf = np.expand_dims(img_array_tf, axis=0)
    img_tensor_pt = pytorch_transforms(image).unsqueeze(0)

    with st.spinner('Calculating parallel model execution...'):
        #  Model 1: Multi-Head TF 
        preds_v1 = v1_model.predict(img_array_tf, verbose=0)
        v1_crop = V1_CROP_NAMES[np.argmax(preds_v1[0])]
        v1_disease = V1_DISEASE_NAMES[np.argmax(preds_v1[1])]
        v1_conf = np.max(preds_v1[0])

        # Model 2: Unified TF 
        preds_v2 = v2_model.predict(img_array_tf, verbose=0)
        v2_full_label = v2_labels[np.argmax(preds_v2[0])]
        v2_conf = np.max(preds_v2[0])

        # Model 3: PyTorch CNN 
        with torch.no_grad():
            outputs = pt_model(img_tensor_pt)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            pt_idx = torch.argmax(probabilities).item()
            pt_conf = probabilities[pt_idx].item()
            pt_full_label = pt_classes[str(pt_idx)]

        #  Model 4: CLIP Foundation Layer with "Unknown" Safeguard 
        clip_target_classes = list(v2_labels) + ['unknown_or_other']
        clip_label, clip_conf = run_clip_ensemble_inference(image, clip_target_classes)

        #  NEW Model 5: YOLO Classification Engine 
        # Ultralytics handles internal scaling automatically on PIL structures
        results = yolo_model(image)

        # 1. Initialize default values in case nothing is detected
        yolo_label = "No Features Localized"
        yolo_conf = 0.0
        annotated_rgb = None

    # 2. Check if YOLO found classification probabilities safely using results[0].probs
    if results and results[0].probs is not None:
        first_result = results[0]  # Grab the actual prediction result object
        
        # Extract the classification class index with the highest probability score
        best_class_idx = first_result.probs.top1
        yolo_label = first_result.names[best_class_idx]
        yolo_conf = first_result.probs.top1conf.item()
        
        # For classification, there are no bounding boxes to plot. 
        # We can pass the input image forward for the UI layout blocks.
        annotated_image = image
        # st.image(annotated_image, caption=f"YOLOv8 Class Match: {format_disease_name(yolo_label)} ({yolo_conf:.2%})", use_container_width=True)
    else:
        st.info("YOLO Classification Engine was unable to process the asset matrix.")

    #  UPDATED COMPARISON MATRIX GRID ---

    st.markdown("### 📊 Pipeline Analysis Results")
    
    # Process text output mappings uniformly
    v1_clean = f"{v1_crop.title()} - {v1_disease}"
    v2_clean = format_disease_name(v2_full_label)
    v3_clean = format_disease_name(pt_full_label)
    clip_clean = format_disease_name(clip_label)
    yolo_clean = format_disease_name(yolo_label)

    # Output grid featuring YOLO as top tier object localizer
    comparison_matrix = f"""
    | Pipeline Model | Framework Architecture | Predicted Classification | Confidence Score |
    | :--- | :--- | :--- | :--- |
    | **YOLO Object Detect** | Ultralytics Bounding Layer | `{yolo_clean}` | **{yolo_conf:.1%}** |
    | **CLIP Foundation** | OpenAI ViT Semantic Guard | `{clip_clean}` | **{clip_conf:.1%}** |
    | **V3: Custom CNN** | PyTorch Residual Network | `{v3_clean}` | **{pt_conf:.1%}** |
    | **V2: Unified Model** | TensorFlow Keras Single-Head | `{v2_clean}` | **{v2_conf:.1%}** |
    | **V1: Multi-Head Model** | TensorFlow Keras Multi-Head | `{v1_clean}` | **{v1_conf:.1%}** |
    """
    st.markdown(comparison_matrix)
    
    st.markdown("---")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    
    #  VISUAL BOUNDING-BOX RENDERING 
   
    if annotated_rgb is not None:
        st.markdown("###  YOLO Localized Feature Mapping")
        st.image(annotated_rgb, caption="YOLO Bounding Box Localization (Pustules/Lesions Map)", use_container_width=True)
        st.markdown("---")
    

    #  MODEL BREAKDOWN CONTAINERS 

    st.markdown("### 🔍 Model Breakdown Overview")
    
    # Row 0: YOLO
    with st.container():
        st.markdown(f"####  Ultralytics YOLO Localization Framework")
        st.markdown(f"**Top Detected Object Patch:** `{yolo_clean}` &nbsp;|&nbsp; **Local Feature Certainty:** `{yolo_conf:.1%}`")
        st.caption("Status: Active localization layer. Explicitly maps coordinates of tissue discoloration to target labels.")

    st.markdown("---")
    
    # Row 1: CLIP
    with st.container():
        st.markdown(f"####  OpenAI CLIP Foundation Model")
        st.markdown(f"**Identified Target:** `{clip_clean}` &nbsp;|&nbsp; **System Confidence:** `{clip_conf:.1%}`")
        st.caption("Status: Active semantic verification layer. Disregards background soil and lighting noise by mapping textures to linguistic concepts.")
    
    st.markdown("---")
    
    # Row 2: PyTorch CNN
    with st.container():
        st.markdown(f"####  V3: Custom PyTorch CNN")
        st.markdown(f"**Identified Target:** `{v3_clean}` &nbsp;|&nbsp; **System Confidence:** `{pt_conf:.1%}`")
        st.caption("Status: Local Custom Network. Evaluates localized pixel clusters; susceptible to fine-grain texture confusion.")

    st.markdown("---")

    # Row 3: Unified TF
    with st.container():
        st.markdown(f"####  V2: Unified Keras Model")
        st.markdown(f"**Identified Target:** `{v2_clean}` &nbsp;|&nbsp; **System Confidence:** `{v2_conf:.1%}`")
        st.caption("Status: Legacy Unified Framework. Single-head dense layer projection.")

    st.markdown("---")

    # Row 4: Multi-Head TF
    with st.container():
        st.markdown("####  V1: Multi-Head Keras Model")
        v1_matrix = f"""
        | Segment Metric | Extracted Prediction | Feature Certainty |
        | :--- | :--- | :--- |
        | **Inferred Crop Type** | `{v1_crop.title()}` | {v1_conf:.1%} |
        | **Inferred Condition** | `{v1_disease}` | {v1_conf:.1%} |
        """
        st.markdown(v1_matrix)

    # Bottom Insight Note
    st.markdown("---")
    st.info("**Architectural Insight:** Cross-referencing YOLO's localized region bounding maps with CLIP's linguistic semantic classifications creates a robust framework validation safety net for edge deployments.")