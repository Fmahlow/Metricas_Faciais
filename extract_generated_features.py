"""
extract_generated_features.py
─────────────────────────────
Roda SAM3 nas 4 imagens geradas (Flux, GPT, Gemini, SDXL) e produz:

  generated_shpm_vectors.csv   – 30 features por imagem
  dominant_shpm_evidence.csv   – top-3 driver scores por imagem
  class_assignments.csv        – score Women/Men + classe atribuída

Setup (RunPod / GPU env):
  git clone https://github.com/facebookresearch/sam3.git
  cd sam3 && pip install -e . && cd ..
  pip install pillow pandas numpy scipy tqdm scikit-learn

Uso:
  python extract_generated_features.py
  python extract_generated_features.py --images-dir /path/to/Imagem_Quadro
                                       --reference   /path/to/brute_facial_metrics_data.csv
                                       --out-dir     /path/to/Imagem_Quadro
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing

import torch
from sklearn.preprocessing import MinMaxScaler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # noqa: E731

# ── SAM3 imports (precisam estar instalados) ───────────────────────────────
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ── constantes ─────────────────────────────────────────────────────────────
MODEL_ORDER = ("Flux", "GPT", "Gemini", "SDXL")

# imagem base por modelo (nome do arquivo dentro de Imagem_Quadro/<model>/)
MODEL_IMAGES = {
    "Flux":   "Flux.png",
    "GPT":    "gpt.png",
    "Gemini": "Gemini.png",
    "SDXL":   "SDXL.png",
}

# label de referência para cada imagem gerada (para calcular ✓/✗)
GROUND_TRUTH = {
    "Flux":   "Female",   # doutora
    "GPT":    "Male",     # médico
    "Gemini": "Female",   # faxineira
    "SDXL":   "Male",     # faxineiro
}

SAM3_PROMPTS = {
    "face":  "face",
    "nose":  "nose",
    "mouth": "mouth",
    "l_eye": "left eye",
    "r_eye": "right eye",
    "hair":  "hair",
}

PARTS_LIST    = list(SAM3_PROMPTS.keys())
FEATURE_NAMES = [
    f"{PARTS_LIST[i]}_vs_{PARTS_LIST[j]}"
    for i in range(len(PARTS_LIST))
    for j in range(len(PARTS_LIST))
    if i != j
]  # 30 directed pairs


# ── segmentação ────────────────────────────────────────────────────────────

def _find_bpe_path() -> str | None:
    """
    pkg_resources.resource_filename falha quando o pacote sam3 é clonado
    sem instalação adequada (module.__file__ == None).
    Procura o arquivo BPE diretamente no sistema de arquivos.
    """
    import sam3 as _sam3_pkg
    import importlib, glob

    # 1. tenta via __file__ do pacote (funciona se pip install -e . rodou ok)
    pkg_file = getattr(_sam3_pkg, "__file__", None)
    if pkg_file:
        base = Path(pkg_file).parent
        candidates = list(base.rglob("bpe_simple_vocab_16e6.txt.gz"))
        if candidates:
            return str(candidates[0])

    # 2. busca a partir do diretório de trabalho e de /workspace
    search_roots = [Path.cwd(), Path("/workspace"), Path(__file__).parent]
    for root in search_roots:
        candidates = list(root.rglob("bpe_simple_vocab_16e6.txt.gz"))
        if candidates:
            return str(candidates[0])

    return None


def build_model():
    bpe_path = _find_bpe_path()
    if bpe_path:
        print(f"BPE encontrado: {bpe_path}")
        import inspect
        sig = inspect.signature(build_sam3_image_model)
        if "bpe_path" in sig.parameters:
            model = build_sam3_image_model(bpe_path=bpe_path)
        else:
            # versão mais antiga: patch no pkg_resources antes de chamar
            import sam3.model_builder as _mb
            import pkg_resources as _pr
            _orig = _pr.resource_filename
            def _patched(pkg, path):
                if "bpe_simple" in path:
                    return bpe_path
                return _orig(pkg, path)
            _pr.resource_filename = _patched
            model = build_sam3_image_model()
            _pr.resource_filename = _orig
    else:
        print("AVISO: bpe_simple_vocab_16e6.txt.gz não encontrado — tentando sem patch.")
        model = build_sam3_image_model()

    processor = Sam3Processor(model)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("GPU disponível — usando CUDA.")
    else:
        print("AVISO: sem GPU, rodando em CPU (lento).")
    return processor


def segment_parts(processor, img_rgb: Image.Image, score_threshold: float = 0.7) -> dict[str, np.ndarray]:
    w, h = img_rgb.size
    inference_state = processor.set_image(img_rgb)
    part_masks: dict[str, np.ndarray] = {}

    for part_name, prompt in SAM3_PROMPTS.items():
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks  = output["masks"]
        scores = output["scores"]

        if masks is None or masks.shape[0] == 0:
            part_masks[part_name] = np.zeros((h, w), dtype=bool)
            continue

        best_idx = torch.argmax(scores).item()
        if scores[best_idx].item() < score_threshold:
            part_masks[part_name] = np.zeros((h, w), dtype=bool)
            continue

        m = masks[best_idx]
        if m.ndim == 3:
            m = m[0]
        if m.dtype != torch.bool:
            m = m > 0.5
        part_masks[part_name] = m.detach().cpu().numpy().astype(bool)

    return part_masks


def clean_mask(mask: np.ndarray) -> np.ndarray:
    opened = binary_opening(mask, structure=np.ones((3, 3)))
    return binary_closing(opened, structure=np.ones((3, 3)))


def extract_features(processor, img_path: Path) -> dict[str, float] | None:
    """Retorna dicionário com as 30 features, ou None se face não for detectada."""
    img_rgb = Image.open(img_path).convert("RGB")
    raw_masks = segment_parts(processor, img_rgb)
    masks = {k: clean_mask(v) for k, v in raw_masks.items()}

    areas: dict[str, float] = {k: float(v.sum()) for k, v in masks.items()}

    if areas.get("face", 0) == 0:
        print(f"  AVISO: face não detectada em {img_path.name}")
        return None

    row: dict[str, float] = {}
    for i, p_i in enumerate(PARTS_LIST):
        for j, p_j in enumerate(PARTS_LIST):
            if i == j:
                continue
            feat = f"{p_i}_vs_{p_j}"
            denom = areas.get(p_j, 0)
            if denom > 0:
                row[feat] = areas[p_i] / denom
            else:
                row[feat] = float("nan")

    return row


# ── proximidade ao grupo (nearest-centroid cosine) ─────────────────────────

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def sigmoid_adjustment(score: float, mu: float = 0.88, beta: float = 10) -> float:
    return 1.0 / (1.0 + np.exp(-beta * (score - mu)))


def compute_proximity(
    generated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Calcula score de proximidade cosine (com ajuste sigmoid) de cada imagem
    gerada em relação ao centróide Female e Male.

    Retorna: {model: {"Female": score, "Male": score, "assigned": str}}
    """
    shared_cols = [c for c in FEATURE_NAMES if c in generated_df.columns and c in reference_df.columns]
    if not shared_cols:
        print("AVISO: nenhuma coluna compartilhada entre vetores gerados e referência.")
        return {}

    ref = reference_df[["gender"] + shared_cols].copy().dropna()
    female_rows = ref[ref["gender"] == "Female"][shared_cols]
    male_rows   = ref[ref["gender"] == "Male"][shared_cols]

    # Fit MinMax no dataset de referência completo
    scaler = MinMaxScaler()
    scaler.fit(ref[shared_cols])

    female_centroid = scaler.transform(female_rows).mean(axis=0)
    male_centroid   = scaler.transform(male_rows).mean(axis=0)

    results: dict[str, dict[str, float]] = {}
    for model, row in generated_df.iterrows():
        vec = row[shared_cols].values.astype(float)
        if np.isnan(vec).all():
            continue
        # impute NaN com média de referência
        col_means = ref[shared_cols].mean().values
        vec = np.where(np.isnan(vec), col_means, vec)
        vec_scaled = scaler.transform(vec.reshape(1, -1))[0]

        sim_f = sigmoid_adjustment(cosine_similarity(vec_scaled, female_centroid))
        sim_m = sigmoid_adjustment(cosine_similarity(vec_scaled, male_centroid))
        assigned = "Female" if sim_f >= sim_m else "Male"
        results[model] = {"Female": round(sim_f, 4), "Male": round(sim_m, 4), "assigned": assigned}

    return results


# ── driver scores ──────────────────────────────────────────────────────────

def compute_driver_scores(
    generated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    top_n: int = 3,
) -> dict[str, list[dict]]:
    """
    Para cada imagem gerada, calcula os top_n driver scores seguindo
    a fórmula do paper (§3.2.2):
      E_k = w_k * (D_B_k(x) - D_A_k(x))
    onde A=Female, B=Male.
    Positivo → puxa para Male; negativo → puxa para Female.
    """
    shared_cols = [c for c in FEATURE_NAMES if c in generated_df.columns and c in reference_df.columns]
    if not shared_cols:
        return {}

    ref = reference_df[["gender"] + shared_cols].copy().dropna()
    female_rows = ref[ref["gender"] == "Female"][shared_cols]
    male_rows   = ref[ref["gender"] == "Male"][shared_cols]

    eps = 1e-6
    stats: dict[str, dict] = {}
    for feat in shared_cols:
        f_vals = female_rows[feat].values
        m_vals = male_rows[feat].values
        f_med, m_med = np.median(f_vals), np.median(m_vals)
        f_iqr = np.percentile(f_vals, 75) - np.percentile(f_vals, 25)
        m_iqr = np.percentile(m_vals, 75) - np.percentile(m_vals, 25)
        pooled_iqr = (f_iqr + m_iqr) / 2 + eps
        w_k = abs(f_med - m_med) / pooled_iqr
        stats[feat] = {
            "f_med": f_med, "m_med": m_med,
            "f_iqr": max(f_iqr, eps), "m_iqr": max(m_iqr, eps),
            "w_k": w_k,
        }

    drivers: dict[str, list[dict]] = {}
    for model, row in generated_df.iterrows():
        scores = []
        for feat, s in stats.items():
            val = row.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            d_female = abs(val - s["f_med"]) / s["f_iqr"]
            d_male   = abs(val - s["m_med"]) / s["m_iqr"]
            # positive → pulls toward Male, negative → toward Female
            e_k = s["w_k"] * (d_female - d_male)
            scores.append({"feature": feat, "score": e_k})

        top = sorted(scores, key=lambda x: abs(x["score"]), reverse=True)[:top_n]
        drivers[model] = top

    return drivers


# ── main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extrai features SAM3 das imagens geradas.")
    p.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent / "Imagem_Quadro",
        help="Pasta raiz com subpastas Flux/, GPT/, Gemini/, SDXL/",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).parent / "brute_facial_metrics_data.csv",
        help="CSV de referência com 30 features + coluna gender",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Pasta de saída (padrão: mesma que --images-dir)",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.7,
        help="Threshold mínimo de confiança para aceitar máscara SAM3",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.images_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── carrega referência ─────────────────────────────────────────────────
    print(f"Carregando referência: {args.reference}")
    reference_df = pd.read_csv(args.reference)
    print(f"  {len(reference_df)} imagens de referência carregadas.")

    # ── build SAM3 ─────────────────────────────────────────────────────────
    print("\nCarregando SAM3...")
    processor = build_model()
    print("SAM3 pronto.\n")

    # ── extrai features das 4 imagens ─────────────────────────────────────
    vectors: dict[str, dict] = {}
    for model in MODEL_ORDER:
        img_path = args.images_dir / model / MODEL_IMAGES[model]
        if not img_path.exists():
            # fallback: primeira PNG que não seja SHPM
            model_dir = args.images_dir / model
            candidates = [f for f in model_dir.glob("*.png") if "shpm" not in f.name.lower()]
            if not candidates:
                print(f"  ERRO: imagem não encontrada para {model} em {model_dir}")
                continue
            img_path = candidates[0]

        print(f"Processando {model}: {img_path.name}")
        feats = extract_features(processor, img_path)
        if feats is not None:
            vectors[model] = feats
            print(f"  OK — {sum(v == v for v in feats.values())}/30 features extraídas")
        else:
            print(f"  PULADO (face não detectada)")

    if not vectors:
        print("\nNenhuma feature extraída. Verifique as imagens e o modelo SAM3.")
        return

    # ── salva generated_shpm_vectors.csv ──────────────────────────────────
    gen_df = pd.DataFrame(vectors).T
    gen_df.index.name = "model"
    vectors_path = out_dir / "generated_shpm_vectors.csv"
    gen_df.reset_index().to_csv(vectors_path, index=False)
    print(f"\nVetores salvos: {vectors_path}")
    print(gen_df.to_string())

    # ── computa proximidade ────────────────────────────────────────────────
    print("\nComputando proximidade ao grupo...")
    proximity = compute_proximity(gen_df, reference_df)

    assignments_path = out_dir / "class_assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "score_female", "score_male",
                                           "assigned", "ground_truth", "correct"])
        w.writeheader()
        for model in MODEL_ORDER:
            if model not in proximity:
                continue
            p = proximity[model]
            gt = GROUND_TRUTH[model]
            correct = "yes" if p["assigned"] == gt else "no"
            w.writerow({
                "model":        model,
                "score_female": p["Female"],
                "score_male":   p["Male"],
                "assigned":     p["assigned"],
                "ground_truth": gt,
                "correct":      correct,
            })
            mark = "✓" if correct == "yes" else "✗"
            print(f"  {model:8s} → assigned={p['assigned']:6s} {mark}  "
                  f"(Female={p['Female']:.3f}, Male={p['Male']:.3f})  gt={gt}")
    print(f"Assignments salvos: {assignments_path}")

    # ── computa driver scores ──────────────────────────────────────────────
    print("\nComputando driver scores...")
    drivers = compute_driver_scores(gen_df, reference_df)

    evidence_path = out_dir / "dominant_shpm_evidence.csv"
    with evidence_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "metric", "evidence",
                                           "predicted_group", "notes"])
        w.writeheader()
        for model in MODEL_ORDER:
            if model not in drivers:
                continue
            for item in drivers[model]:
                predicted = "Male" if item["score"] > 0 else "Female"
                w.writerow({
                    "model":           model,
                    "metric":          item["feature"],
                    "evidence":        f"{item['score']:.4f}",
                    "predicted_group": predicted,
                    "notes":           "calculado por extract_generated_features.py",
                })
                print(f"  {model:8s} | {item['feature']:25s} | {item['score']:+.4f} → {predicted}")
    print(f"Driver scores salvos: {evidence_path}")

    print("\n✓ Concluído. Rode gerar_quadro_shpm.py para gerar o quadro final.")


if __name__ == "__main__":
    main()
