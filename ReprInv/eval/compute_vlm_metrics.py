import re
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

_vlm = None
_processor = None
_embed_model = None


def _load_models(device: str = "cuda"):
    global _vlm, _processor, _embed_model
    if _vlm is None:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        _processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        _vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
        )
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _vlm, _processor, _embed_model

def _generate(model, processor, messages, max_new_tokens: int = 512) -> str:
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )
    response = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del inputs, out, image_inputs, video_inputs
    return response.strip()


# Prompts
def _object_prompt(image: Image.Image) -> list:
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": (
            "List up to ten objects in this image.\n\n"
            "Include: furniture, electronics, appliances, vehicles, decorations, "
            "plants, signs, equipment, distinctive elements in a building.\n"
            "Exclude: general structural elements (wall, floor, ceiling, window, "
            "door, sky, tree, building, etc.).\n\n"
            "Use simple names: \"chair\" not \"green office chair\".\n"
            "Ignore colors and materials.\n\nOnly output a numbered list:\n1.\n2.\n..."
        )},
    ]}]

def _texture_prompt(image: Image.Image) -> list:
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": (
            "List up to ten distinctive details and textures visible in this image "
            "that would allow you to recognize it among images with similar layouts. "
            "Be precise and specific.\n\n"
            "Exclude: general structural elements (wall, floor, ceiling, window, "
            "door, sky, tree, building, etc.).\nIgnore colors.\n\n"
            "Output a numbered list:\n1.\n2.\n..."
        )},
    ]}]

def _scene_desc_prompt(image: Image.Image) -> list:
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": (
            "Describe this image in a few sentences. Focus on the spatial layout, "
            "distinctive features, and unique elements that would help identify this "
            "specific location."
        )},
    ]}]

def _parse_list(text: str) -> List[str]:
    _SKIP = {"note:", "however,", "please note", "based on the image",
             "i can see", "the image shows", "here are", "here is"}
    items, seen = [], set()
    for line in text.strip().split("\n"):
        line = re.sub(r"^\s*(?:\d+[\)\.\:\-]|[-•*])\s*", "", line).strip()
        if len(line) < 2:
            continue
        if any(s in line.lower() for s in _SKIP):
            continue
        key = line.lower()
        if key not in seen:
            seen.add(key)
            items.append(line.rstrip(".").strip())
    return items

def _match_items(
    items_gt: List[str],
    items_pred: List[str],
    embed_model,
    threshold: float = 0.5,
) -> Tuple[float, int, int, list]:
    if not items_gt or not items_pred:
        return 0.0, 0, max(len(items_gt), 1), []
    emb_gt   = embed_model.encode(items_gt,   normalize_embeddings=True)
    emb_pred = embed_model.encode(items_pred, normalize_embeddings=True)
    sim = emb_gt @ emb_pred.T
    rows, cols = linear_sum_assignment(1.0 - sim)
    matches = [(items_gt[r], items_pred[c], float(sim[r, c]))
               for r, c in zip(rows, cols) if sim[r, c] >= threshold]
    return len(matches) / max(len(items_gt), 1), len(matches), len(items_gt), matches


def _scene_similarity(desc_gt: str, desc_pred: str, embed_model) -> float:
    emb = embed_model.encode([desc_gt, desc_pred], normalize_embeddings=True)
    return float(emb[0] @ emb[1])


def compute_vlm_metrics(
    recons_ims: list,
    gt_ims: list,
    n_samples: int = 10,
    seed: int = 42,
    object_threshold: float = 0.5,
    texture_threshold: float = 0.5,
    device: str = "cuda",
) -> Tuple[dict, list]:
    vlm, processor, embed_model = _load_models(device)

    np.random.seed(seed)
    n = len(recons_ims)
    indices = np.random.choice(n, min(n_samples, n), replace=False)

    per_image = []
    for idx in indices:
        pred_np = recons_ims[idx]
        gt_np   = gt_ims[idx]

        pred_pil = Image.fromarray((pred_np * 255).round().clip(0, 255).astype(np.uint8))
        gt_pil   = Image.fromarray(gt_np if gt_np.dtype == np.uint8
                                   else (gt_np * 255).astype(np.uint8))

        # Object recovery
        gt_obj_raw   = _generate(vlm, processor, _object_prompt(gt_pil))
        pred_obj_raw = _generate(vlm, processor, _object_prompt(pred_pil))
        gt_obj   = _parse_list(gt_obj_raw)
        pred_obj = _parse_list(pred_obj_raw)
        or_score, or_n, or_total, or_matches = _match_items(gt_obj, pred_obj, embed_model, object_threshold)

        # Texture recovery
        gt_tex_raw   = _generate(vlm, processor, _texture_prompt(gt_pil))
        pred_tex_raw = _generate(vlm, processor, _texture_prompt(pred_pil))
        gt_tex   = _parse_list(gt_tex_raw)
        pred_tex = _parse_list(pred_tex_raw)
        tr_score, tr_n, tr_total, tr_matches = _match_items(gt_tex, pred_tex, embed_model, texture_threshold)

        # Scene recognition
        gt_desc   = _generate(vlm, processor, _scene_desc_prompt(gt_pil))
        pred_desc = _generate(vlm, processor, _scene_desc_prompt(pred_pil))
        sr_score  = _scene_similarity(gt_desc, pred_desc, embed_model)

        # Log VLM outputs
        generations = {
            "objects": {
                "gt_raw": gt_obj_raw, "pred_raw": pred_obj_raw,
                "gt_items": gt_obj, "pred_items": pred_obj,
                "matches": [{"gt": g, "pred": p, "similarity": s} for g, p, s in or_matches],
                "n_matched": or_n, "n_gt": or_total,
            },
            "textures": {
                "gt_raw": gt_tex_raw, "pred_raw": pred_tex_raw,
                "gt_items": gt_tex, "pred_items": pred_tex,
                "matches": [{"gt": g, "pred": p, "similarity": s} for g, p, s in tr_matches],
                "n_matched": tr_n, "n_gt": tr_total,
            },
            "scene": {"gt_raw": gt_desc, "pred_raw": pred_desc},
        }

        per_image.append({"image_idx": int(idx), "or": or_score, "tr": tr_score,
                          "sr": sr_score, "generations": generations})

    def _agg(key):
        vals = [m[key] for m in per_image]
        return float(np.mean(vals)), float(np.std(vals))

    or_m, or_s = _agg("or")
    tr_m, tr_s = _agg("tr")
    sr_m, sr_s = _agg("sr")

    agg = {
        "OR_mean": or_m, "OR_std": or_s,
        "TR_mean": tr_m, "TR_std": tr_s,
        "SR_mean": sr_m, "SR_std": sr_s,
        "num_images": len(per_image),
    }
    return agg, per_image
