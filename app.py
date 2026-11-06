
import json
import math
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import BlipProcessor, BlipForConditionalGeneration


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VOCAB_PATH = BASE_DIR / "notebook" / "vocabulary.json"

# Preserve the existing experimental checkpoint
CHECKPOINT_PATH = (
    BASE_DIR
    / "notebook"
    / "checkpoints_vit_experiment"
    / "vit_transformer_experiment.pth"
)

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

DEVICE = torch.device("cpu")
MAX_LENGTH = 38
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered",
)


# ============================================================
# 2. CUSTOM TRANSFORMER DECODER
# ============================================================

class TransformerCaptionDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        visual_dim=768,
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        feedforward_dim=1024,
        max_length=38,
        dropout=0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_length = max_length

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.visual_projection = nn.Linear(
            visual_dim,
            embed_dim,
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embed_dim,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(
            embed_dim,
            vocab_size,
        )

    def forward(self, visual_features, caption_ids):

        batch_size, caption_length = caption_ids.shape

        if caption_length > self.max_length:
            raise ValueError(
                f"Caption length {caption_length} exceeds "
                f"the model limit of {self.max_length}."
            )

        memory = self.visual_projection(visual_features)

        positions = torch.arange(
            caption_length,
            device=caption_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        target = (
            self.embedding(caption_ids)
            * math.sqrt(self.embed_dim)
            + self.position_embedding(positions)
        )

        causal_mask = torch.triu(
            torch.ones(
                caption_length,
                caption_length,
                device=caption_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        padding_mask = caption_ids.eq(0)

        decoded = self.transformer_decoder(
            tgt=target,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        return self.output_layer(decoded)


# ============================================================
# 3. LOAD CUSTOM VIT + TRANSFORMER MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_captioning_model():

    if not VOCAB_PATH.is_file():
        raise FileNotFoundError(
            f"Vocabulary file not found:\n{VOCAB_PATH}"
        )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Custom model checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocab_data = json.load(file)

    if not isinstance(vocab_data, dict):
        raise ValueError(
            "Vocabulary must be a word-to-ID dictionary."
        )

    for token in ["<PAD>", "<START>", "<END>", "<UNK>"]:
        if token not in vocab_data:
            raise ValueError(
                f"Required vocabulary token is missing: {token}"
            )

    id_to_word = {
        int(index): word
        for word, index in vocab_data.items()
    }

    vocab_size = len(vocab_data)

    weights = ViT_B_16_Weights.DEFAULT

    vit_model = vit_b_16(weights=weights)
    vit_model.heads = nn.Identity()

    for parameter in vit_model.parameters():
        parameter.requires_grad = False

    vit_model = vit_model.to(DEVICE)
    vit_model.eval()

    decoder = TransformerCaptionDecoder(
        vocab_size=vocab_size,
        visual_dim=768,
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        feedforward_dim=1024,
        max_length=MAX_LENGTH,
        dropout=0.1,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if checkpoint.get("vocab_size") != vocab_size:
        raise ValueError(
            "Vocabulary size does not match the checkpoint."
        )

    decoder.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    decoder = decoder.to(DEVICE)
    decoder.eval()

    model_info = {
        "epoch": checkpoint.get("epoch", "Unknown"),
        "val_loss": checkpoint.get("val_loss"),
        "vocab_size": vocab_size,
    }

    return (
        vit_model,
        decoder,
        weights.transforms(),
        id_to_word,
        vocab_data,
        model_info,
    )


# ============================================================
# 4. LOAD BLIP MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_blip_model():

    processor = BlipProcessor.from_pretrained(
        BLIP_MODEL_NAME
    )

    model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL_NAME
    )

    model = model.to(DEVICE)
    model.eval()

    return processor, model


# ============================================================
# 5. EXTRACT VIT PATCH FEATURES
# ============================================================

def extract_vit_patch_tokens(image_tensor, vit_model):

    with torch.inference_mode():

        x = vit_model._process_input(image_tensor)

        batch_size = x.shape[0]

        class_token = vit_model.class_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            (class_token, x),
            dim=1,
        )

        x = vit_model.encoder(x)

        # Remove class token: retain 196 image patches
        patch_tokens = x[:, 1:, :]

    return patch_tokens


# ============================================================
# 6. CUSTOM MODEL: GREEDY CAPTION GENERATION
# ============================================================

def generate_caption(
    visual_features,
    decoder,
    id_to_word,
    vocab_data,
):

    pad_id = int(vocab_data["<PAD>"])
    start_id = int(vocab_data["<START>"])
    end_id = int(vocab_data["<END>"])

    caption_ids = [start_id]

    decoder.eval()

    with torch.inference_mode():

        for _ in range(MAX_LENGTH - 1):

            input_ids = torch.tensor(
                [caption_ids],
                dtype=torch.long,
                device=DEVICE,
            )

            logits = decoder(
                visual_features,
                input_ids,
            )

            next_token = int(
                logits[0, -1].argmax(dim=-1).item()
            )

            if next_token == end_id:
                break

            caption_ids.append(next_token)

    words = []

    for token_id in caption_ids[1:]:

        if token_id in (pad_id, start_id, end_id):
            continue

        word = id_to_word.get(token_id, "<UNK>")

        if word not in ("<PAD>", "<START>", "<END>"):
            words.append(word)

    return " ".join(words).strip()


# ============================================================
# 7. CUSTOM MODEL: BEAM SEARCH
# ============================================================

def generate_multiple_captions(
    visual_features,
    decoder,
    id_to_word,
    vocab_data,
    num_captions=3,
    beam_width=5,
):

    pad_id = int(vocab_data["<PAD>"])
    start_id = int(vocab_data["<START>"])
    end_id = int(vocab_data["<END>"])

    decoder.eval()

    # Each beam: (token IDs, cumulative log probability, finished)
    beams = [([start_id], 0.0, False)]

    with torch.inference_mode():

        for _ in range(MAX_LENGTH - 1):

            candidates = []

            for token_ids, score, finished in beams:

                if finished:
                    candidates.append(
                        (token_ids, score, True)
                    )
                    continue

                input_ids = torch.tensor(
                    [token_ids],
                    dtype=torch.long,
                    device=DEVICE,
                )

                logits = decoder(
                    visual_features,
                    input_ids,
                )

                next_token_logits = logits[0, -1]

                log_probs = torch.log_softmax(
                    next_token_logits,
                    dim=-1,
                )

                log_probs[pad_id] = float("-inf")
                log_probs[start_id] = float("-inf")

                top_tokens = torch.topk(
                    log_probs,
                    k=min(beam_width, log_probs.numel()),
                )

                for token_log_prob, token_id in zip(
                    top_tokens.values.tolist(),
                    top_tokens.indices.tolist(),
                ):

                    new_ids = token_ids + [int(token_id)]
                    new_score = score + float(token_log_prob)

                    candidates.append(
                        (
                            new_ids,
                            new_score,
                            int(token_id) == end_id,
                        )
                    )

            candidates.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            beams = candidates[:beam_width]

            if all(finished for _, _, finished in beams):
                break

    captions = []
    seen = set()

    for token_ids, score, _ in sorted(
        beams,
        key=lambda item: item[1],
        reverse=True,
    ):

        words = []

        for token_id in token_ids[1:]:

            if token_id == end_id:
                break

            if token_id in (pad_id, start_id):
                continue

            word = id_to_word.get(token_id, "<UNK>")

            if word != "<UNK>" and not word.startswith("<"):
                words.append(word)

        caption = " ".join(words).strip()
        normalized = caption.lower()

        if caption and normalized not in seen:
            captions.append(caption)
            seen.add(normalized)

        if len(captions) >= num_captions:
            break

    return captions


# ============================================================
# 8. BLIP CAPTION GENERATION
# ============================================================

def generate_blip_caption(image, processor, model):

    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
        )

    caption = processor.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return caption.strip()


# ============================================================
# 9. STREAMLIT USER INTERFACE
# ============================================================

st.title("🖼️ AI Image Caption Generator")

st.write(
    "Upload an image and compare a custom ViT + Transformer "
    "captioning model with pretrained BLIP."
)

st.caption(
    "M.Tech Data Science Project | Phase 8 | CPU inference"
)

st.divider()

selected_model = st.selectbox(
    "Choose captioning model",
    [
        "Custom ViT + Transformer",
        "BLIP (Pretrained)",
    ],
)

if selected_model == "Custom ViT + Transformer":

    with st.expander("About the custom model"):

        st.write(
            "ViT-B/16 extracts image patch features, and a "
            "Transformer decoder trained on a Flickr8k-style "
            "dataset generates captions using beam search."
        )

        st.write(
            "The model may produce inaccurate descriptions "
            "for unfamiliar images."
        )

else:

    with st.expander("About BLIP"):

        st.write(
            "BLIP is a pretrained vision-language model that "
            "generates captions using learned image-text "
            "relationships."
        )

        st.write(
            "The first load may download pretrained model "
            "files from Hugging Face. BLIP can also make "
            "incorrect predictions."
        )

uploaded_file = st.file_uploader(
    "Choose an image",
    type=SUPPORTED_FORMATS,
    help="Supported formats: JPG, JPEG, PNG and WEBP.",
)

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)
        image.load()
        image = image.convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

        generate_button = st.button(
            "Generate Caption(s)",
            type="primary",
            use_container_width=True,
        )

        if generate_button:

            if selected_model == "BLIP (Pretrained)":

                with st.spinner(
                    "Loading BLIP and generating caption..."
                ):

                    processor, blip_model = load_blip_model()

                    caption = generate_blip_caption(
                        image,
                        processor,
                        blip_model,
                    )

                st.subheader("BLIP Caption")

                if caption:
                    st.write(caption)
                else:
                    st.warning("BLIP returned an empty caption.")

                st.caption(
                    "BLIP predictions may contain incorrect "
                    "objects or scene details."
                )

                with st.expander("Model information"):

                    st.write(
                        f"Model: {BLIP_MODEL_NAME}"
                    )

                    st.write("Inference device: CPU")

            else:

                with st.spinner(
                    "Loading the custom model and generating captions..."
                ):

                    (
                        vit_model,
                        decoder,
                        transform,
                        id_to_word,
                        vocab_data,
                        model_info,
                    ) = load_captioning_model()

                    image_tensor = transform(
                        image
                    ).unsqueeze(0).to(DEVICE)

                    visual_features = extract_vit_patch_tokens(
                        image_tensor,
                        vit_model,
                    )

                    if tuple(visual_features.shape) != (
                        1, 196, 768
                    ):
                        raise RuntimeError(
                            "Unexpected visual feature shape: "
                            f"{tuple(visual_features.shape)}"
                        )

                    captions = generate_multiple_captions(
                        visual_features,
                        decoder,
                        id_to_word,
                        vocab_data,
                        num_captions=3,
                        beam_width=5,
                    )

                st.subheader(
                    "Custom ViT + Transformer Captions"
                )

                if captions:

                    for i, caption in enumerate(
                        captions,
                        start=1,
                    ):

                        st.markdown(f"**Caption {i}:**")
                        st.write(caption)
                        st.divider()

                else:

                    st.warning(
                        "No caption could be generated. "
                        "Try another image."
                    )

                with st.expander("Model information"):

                    st.write(
                        f"Checkpoint epoch: {model_info['epoch']}"
                    )

                    st.write(
                        f"Vocabulary size: "
                        f"{model_info['vocab_size']}"
                    )

                    val_loss = model_info["val_loss"]

                    if val_loss is not None:
                        st.write(
                            f"Saved validation loss: {val_loss:.4f}"
                        )

                    st.write(
                        "Image feature shape: (1, 196, 768)"
                    )

                    st.write("Inference device: CPU")

    except (UnidentifiedImageError, OSError):

        st.error(
            "The uploaded file could not be read as a valid "
            "image. Please try another JPG, PNG or WEBP file."
        )

    except Exception as error:

        st.error(
            "Caption generation failed. Check the technical "
            "details below."
        )

        with st.expander("Technical error details"):
            st.exception(error)

else:

    st.info("Upload an image to get started.")

st.divider()

st.caption(
    "Caption quality depends on the selected model. "
    "Generated captions are not guaranteed to describe "
    "every image accurately."
)
