\# Image Captioning Using Vision Transformer and Transformer Decoder



An M.Tech Data Science project that generates natural-language descriptions for images using a custom Vision Transformer (ViT) and Transformer Decoder architecture.



\## Project Overview



Image captioning combines Computer Vision and Natural Language Processing to automatically describe the contents of an image.



This project implements and evaluates two deep-learning architectures:



1\. \*\*Baseline Model:\*\* ResNet50 + LSTM

2\. \*\*Proposed Custom Model:\*\* Vision Transformer (ViT-B/16) + Custom Transformer Decoder



The project also includes visual-consistency analysis, hallucination analysis, caption confidence evaluation, error analysis, and a Streamlit web application.



\## Proposed Model Architecture



\- \*\*Visual Encoder:\*\* Pretrained Vision Transformer (ViT-B/16)

\- \*\*Input Image Size:\*\* 224 × 224

\- \*\*Patch Size:\*\* 16 × 16

\- \*\*Image Patches:\*\* 196

\- \*\*Visual Feature Dimension:\*\* 768

\- \*\*Transformer Decoder:\*\* 3 layers

\- \*\*Attention Heads:\*\* 8

\- \*\*Embedding Dimension:\*\* 256

\- \*\*Feed-Forward Dimension:\*\* 1024

\- \*\*Maximum Caption Length:\*\* 38 tokens



The pretrained visual encoder is frozen, while the custom Transformer decoder is trained for image caption generation.



\## Dataset



The project uses a Flickr8k-style image-caption dataset.



\- 8,092 images

\- 40,460 caption records

\- 30,000 training captions

\- 5,000 validation captions

\- 5,000 testing captions



The original dataset is not included in this repository. Please obtain it separately and follow its applicable usage terms.



\## Model Performance



| Evaluation Metric | ResNet50 + LSTM | ViT + Transformer |

|---|---:|---:|

| BLEU-1 | 0.4704 | 0.5241 |

| BLEU-2 | 0.3090 | 0.3615 |

| BLEU-3 | 0.1942 | 0.2409 |

| BLEU-4 | 0.1247 | 0.1583 |

| METEOR | 0.3483 | 0.4055 |

| ROUGE-L | 0.3821 | 0.4149 |



The custom ViT + Transformer model achieved better scores across the reported evaluation metrics.



\## Project Features



\- Image caption generation using the custom ViT + Transformer model.

\- Greedy decoding for caption generation.

\- Beam Search for generating multiple caption candidates.

\- Comparison with the ResNet50 + LSTM baseline.

\- Comparison with the pretrained BLIP model.

\- Hallucination and visual-consistency analysis.

\- Caption confidence and error analysis.

\- Interactive Streamlit web application.



\## Repository Structure



```text

image-captioning-vit-transformer/

│

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

├── .gitattributes

├── vocabulary.json

│

├── notebook/

│   ├── vocabulary.json

│   └── checkpoints\_vit\_experiment/

│       └── vit\_transformer\_experiment.pth

│

├── notebooks/

│   └── Project Jupyter Notebooks

│

├── results/

│   ├── baseline\_metrics.json

│   ├── vit\_transformer\_metrics.json

│   ├── phase4/

│   ├── phase5/

│   └── phase6/

│

├── docs/

├── dataset/

└── sample\_outputs/

```



\## Installation and Execution



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd image-captioning-vit-transformer

```



\### 2. Create and activate a virtual environment



```bash

python -m venv venv

```



Windows:



```bash

venv\\Scripts\\activate

```



\### 3. Install CPU-compatible PyTorch



```bash

pip install torch==2.13.0+cpu torchvision==0.28.0+cpu --index-url https://download.pytorch.org/whl/cpu

```



\### 4. Install the remaining dependencies



```bash

pip install -r requirements.txt

```



\### 5. Run the Streamlit application



```bash

streamlit run app.py

```



Open the local URL displayed in the terminal.



\*\*Note:\*\* The BLIP model may be downloaded automatically from Hugging Face the first time the application uses it. An internet connection is required for the initial download.



\## Model Checkpoint



The trained custom ViT + Transformer checkpoint is stored at:



`notebook/checkpoints\_vit\_experiment/vit\_transformer\_experiment.pth`



Git LFS is used to manage the large model checkpoint.



Install Git LFS before cloning if it is not already installed:



```bash

git lfs install

git lfs pull

```



\## Technologies Used



\- Python

\- PyTorch

\- Torchvision

\- Vision Transformer (ViT)

\- Transformer Decoder

\- LSTM

\- Hugging Face Transformers

\- BLIP

\- Streamlit

\- Jupyter Notebook

\- Natural Language Processing

\- Computer Vision



\## Project Scope



This project explores how transformer-based visual representations and language decoding can be used to generate meaningful image descriptions. It also studies model performance, visual consistency, caption confidence, and practical deployment through a web interface.



\## Academic Project



\*\*Degree:\*\* M.Tech in Data Science  

\*\*Institution:\*\* College of Engineering Pune (COEP)





