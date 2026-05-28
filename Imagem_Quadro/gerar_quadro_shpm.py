from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


MODEL_ORDER = ("Flux", "GPT", "Gemini", "SDXL")
DEFAULT_OUTPUT_NAME = "quadro_shpm_paper.png"
RATIO_COLUMNS = (
    "Nose_vs_Face",
    "Mouth_vs_Face",
    "Eyes_vs_Face",
    "Nose_vs_Mouth",
    "Eyes_vs_Nose",
    "Eyes_vs_Mouth",
)


def default_input_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    if all((script_dir / model).exists() for model in MODEL_ORDER):
        return script_dir

    for name in ("Imagem_Quadro", "imagem_quadro"):
        candidate = script_dir / name
        if candidate.exists():
            return candidate

    return script_dir / "Imagem_Quadro"


def parse_args() -> argparse.Namespace:
    input_dir = default_input_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Gera um quadro 2x2 com imagens de modelos de IA e os respectivos "
            "graficos SHPM sobrepostos em menor escala."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=input_dir,
        help="Pasta que contem as subpastas Flux, GPT, Gemini e SDXL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=input_dir / DEFAULT_OUTPUT_NAME,
        help="Caminho do arquivo PNG final.",
    )
    parser.add_argument(
        "--overlay-scale",
        type=float,
        default=0.46,
        help="Largura do grafico como fracao da largura da imagem base.",
    )
    parser.add_argument(
        "--overlay-margin",
        type=int,
        default=34,
        help="Margem interna, em pixels, usada para posicionar o grafico.",
    )
    parser.add_argument(
        "--graph-bg-opacity",
        type=float,
        default=0.76,
        help="Opacidade do fundo branco semi-transparente atras do grafico.",
    )
    parser.add_argument(
        "--graph-box-padding",
        type=int,
        default=20,
        help="Respiro interno, em pixels, entre o grafico e a caixa arredondada.",
    )
    parser.add_argument(
        "--graph-box-radius",
        type=int,
        default=0,
        help="Raio dos cantos arredondados da caixa do grafico.",
    )
    parser.add_argument(
        "--evidence-csv",
        type=Path,
        default=input_dir / "dominant_shpm_evidence.csv",
        help=(
            "CSV com colunas model, metric e evidence. Evidence negativo favorece "
            "Female; positivo favorece Male."
        ),
    )
    parser.add_argument(
        "--generated-vectors",
        type=Path,
        default=input_dir / "generated_shpm_vectors.csv",
        help=(
            "CSV opcional com model e proporcoes SHPM das imagens geradas. "
            "Quando existe, o script calcula a metrica dominante."
        ),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=input_dir.parent / "facial_proportions_wiki_imdb.csv",
        help="CSV de referencia com distribuicoes Female/Male.",
    )
    parser.add_argument(
        "--lollipop-limit",
        type=float,
        default=2.0,
        help="Limite simetrico do eixo do lollipop plot.",
    )
    parser.add_argument(
        "--no-lollipop",
        action="store_true",
        help="Nao desenha o lollipop de evidencia local.",
    )
    parser.add_argument(
        "--gutter",
        type=int,
        default=46,
        help="Espacamento, em pixels, entre paineis.",
    )
    parser.add_argument(
        "--outer-margin",
        type=int,
        default=84,
        help="Margem externa, em pixels, do quadro final.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI gravado no PNG e no PDF.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Nao desenha os nomes dos modelos sobre as imagens.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Nao salva uma copia em PDF.",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip().replace(",", ".")
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def median(values: list[float]) -> float:
    return percentile(values, 0.5)


def iqr(values: list[float]) -> float:
    return percentile(values, 0.75) - percentile(values, 0.25)


def safe_scale(value: float, fallback: float = 1.0) -> float:
    if abs(value) < 1e-9:
        return fallback
    return value


def short_metric_label(metric: str) -> str:
    return metric.replace("_vs_", "/").replace("_", " ")


def write_evidence_template(path: Path) -> None:
    if path.exists():
        return
    rows = [
        {
            "model": model,
            "metric": "",
            "evidence": "",
            "predicted_group": "",
            "notes": "evidence < 0 favorece Female; evidence > 0 favorece Male",
        }
        for model in MODEL_ORDER
        for _ in range(3)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("model", "metric", "evidence", "predicted_group", "notes"),
        )
        writer.writeheader()
        writer.writerows(rows)


def read_evidence_csv(path: Path) -> dict[str, list[dict[str, str | float]]]:
    evidence: dict[str, list[dict[str, str | float]]] = {}
    if not path.exists():
        return evidence

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            model = (row.get("model") or row.get("Modelo") or "").strip()
            metric = (row.get("metric") or row.get("ratio") or row.get("feature") or "").strip()
            score = parse_float(row.get("evidence") or row.get("score") or row.get("contribution"))
            if not model or not metric or score is None:
                continue
            evidence.setdefault(model, []).append(
                {
                    "metric": metric,
                    "evidence": score,
                    "predicted_group": (row.get("predicted_group") or "").strip(),
                }
            )
    for model, rows in evidence.items():
        evidence[model] = sorted(
            rows,
            key=lambda item: abs(float(item["evidence"])),
            reverse=True,
        )[:3]
    return evidence


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def calculate_evidence_from_vectors(
    generated_vectors: Path,
    reference_csv: Path,
    output_csv: Path,
) -> dict[str, list[dict[str, str | float]]]:
    if not generated_vectors.exists() or not reference_csv.exists():
        return {}

    reference_rows = read_csv_rows(reference_csv)
    generated_rows = read_csv_rows(generated_vectors)
    if not reference_rows or not generated_rows:
        return {}

    generated_columns = set(generated_rows[0].keys())
    reference_columns = set(reference_rows[0].keys())
    features = [
        column
        for column in RATIO_COLUMNS
        if column in generated_columns and column in reference_columns
    ]
    if not features:
        return {}

    stats: dict[str, dict[str, float]] = {}
    for feature in features:
        female_values: list[float] = []
        male_values: list[float] = []
        for row in reference_rows:
            value = parse_float(row.get(feature))
            gender = (row.get("gender") or "").strip().lower()
            if value is None:
                continue
            if gender == "female":
                female_values.append(value)
            elif gender == "male":
                male_values.append(value)

        if not female_values or not male_values:
            continue

        female_median = median(female_values)
        male_median = median(male_values)
        female_iqr = iqr(female_values)
        male_iqr = iqr(male_values)
        pooled_iqr = safe_scale((female_iqr + male_iqr) / 2, fallback=1.0)
        stats[feature] = {
            "female_median": female_median,
            "male_median": male_median,
            "female_iqr": safe_scale(female_iqr, fallback=pooled_iqr),
            "male_iqr": safe_scale(male_iqr, fallback=pooled_iqr),
            "separability": abs(male_median - female_median) / pooled_iqr,
        }

    evidence: dict[str, list[dict[str, str | float]]] = {}
    output_rows = []
    for row in generated_rows:
        model = (
            row.get("model")
            or row.get("Modelo")
            or row.get("filename")
            or row.get("image")
            or ""
        ).strip()
        if not model:
            continue

        feature_scores = []
        for feature, feature_stats in stats.items():
            value = parse_float(row.get(feature))
            if value is None:
                continue

            distance_to_female = abs(value - feature_stats["female_median"]) / feature_stats["female_iqr"]
            distance_to_male = abs(value - feature_stats["male_median"]) / feature_stats["male_iqr"]
            score = (
                (distance_to_female - distance_to_male)
                * feature_stats["separability"]
            )
            feature_scores.append((feature, score))

        if not feature_scores:
            continue

        top_scores = sorted(feature_scores, key=lambda item: abs(item[1]), reverse=True)[:3]
        evidence[model] = []
        for feature, score in top_scores:
            predicted_group = "Male" if score > 0 else "Female"
            item = {
                "metric": feature,
                "evidence": score,
                "predicted_group": predicted_group,
            }
            evidence[model].append(item)
            output_rows.append(
                {
                    "model": model,
                    "metric": feature,
                    "evidence": f"{score:.4f}",
                    "predicted_group": predicted_group,
                    "notes": "calculado a partir de generated_shpm_vectors.csv",
                }
            )

    if output_rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("model", "metric", "evidence", "predicted_group", "notes"),
            )
            writer.writeheader()
            writer.writerows(output_rows)

    return evidence


def load_evidence_data(args: argparse.Namespace) -> dict[str, list[dict[str, str | float]]]:
    if args.no_lollipop:
        return {}

    evidence = calculate_evidence_from_vectors(
        generated_vectors=args.generated_vectors.resolve(),
        reference_csv=args.reference_csv.resolve(),
        output_csv=args.evidence_csv.resolve(),
    )
    if evidence:
        return evidence

    evidence = read_evidence_csv(args.evidence_csv.resolve())
    if evidence:
        return evidence

    write_evidence_template(args.evidence_csv.resolve())
    return {}



def find_base_image(model_dir: Path, model: str) -> Path:
    direct_candidates = (
        model_dir / f"{model}.png",
        model_dir / f"{model.lower()}.png",
        model_dir / f"{model.upper()}.png",
    )
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    pngs = sorted(model_dir.glob("*.png"))
    for path in pngs:
        lower_name = path.name.lower()
        if "shpm" not in lower_name and not lower_name.startswith("."):
            return path

    raise FileNotFoundError(f"Nenhuma imagem base encontrada em {model_dir}")


def find_graph_image(model_dir: Path) -> Path:
    transparent_graphs = sorted(model_dir.glob("shpm_transparente*.png"))
    if transparent_graphs:
        return transparent_graphs[0]

    bars_only = model_dir / "shpm_bars_only.png"
    if bars_only.exists():
        return bars_only

    raise FileNotFoundError(f"Nenhum grafico SHPM encontrado em {model_dir}")


def trim_graph(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")

    alpha = image.getchannel("A")
    if alpha.getextrema()[0] < 255:
        bbox = alpha.getbbox()
        if bbox:
            return image.crop(bbox)

    white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    diff = ImageChops.difference(image, white_background)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)

    return image


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    width = max(1, width)
    height = max(1, round(image.height * (width / image.width)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def recolor_white_graph_text(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            red, green, blue, alpha = pixels[x, y]
            if alpha > 100 and red > 215 and green > 215 and blue > 215:
                pixels[x, y] = (0, 0, 0, alpha)
    return image


def rounded_rect_points(
    left: float,
    top: float,
    right: float,
    bottom: float,
    radius: float,
    steps: int,
) -> list[tuple[float, float]]:
    radius = min(radius, (right - left) / 2, (bottom - top) / 2)
    points: list[tuple[float, float]] = []
    centers = (
        (right - radius, top + radius, -90, 0),
        (right - radius, bottom - radius, 0, 90),
        (left + radius, bottom - radius, 90, 180),
        (left + radius, top + radius, 180, 270),
    )
    for center_x, center_y, start_angle, end_angle in centers:
        for step in range(steps + 1):
            angle = math.radians(start_angle + (end_angle - start_angle) * step / steps)
            points.append(
                (
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                )
            )
    return points


def draw_dotted_rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    dot_radius: int = 2,
    spacing: int = 13,
) -> None:
    points = rounded_rect_points(*box, radius=radius, steps=24)
    distance_since_dot = 0.0
    previous = points[-1]
    for current in points:
        segment_length = math.dist(previous, current)
        if segment_length == 0:
            previous = current
            continue

        travelled = spacing - distance_since_dot
        while travelled <= segment_length:
            ratio = travelled / segment_length
            x = previous[0] + (current[0] - previous[0]) * ratio
            y = previous[1] + (current[1] - previous[1]) * ratio
            draw.ellipse(
                (
                    x - dot_radius,
                    y - dot_radius,
                    x + dot_radius,
                    y + dot_radius,
                ),
                fill=fill,
            )
            travelled += spacing

        distance_since_dot = (distance_since_dot + segment_length) % spacing
        previous = current


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - width) / 2 - bbox[0]
    y = box[1] + (box[3] - box[1] - height) / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def make_lollipop_plot(
    evidence_rows: list[dict[str, str | float]],
    width: int,
    limit: float,
) -> Image.Image:
    evidence_rows = sorted(
        evidence_rows,
        key=lambda item: abs(float(item["evidence"])),
        reverse=True,
    )[:3]
    row_height = 80
    height = 86 + len(evidence_rows) * row_height + 16
    limit = max(0.1, abs(limit))
    max_score = max(abs(float(item["evidence"])) for item in evidence_rows)
    if max_score > limit:
        limit = max_score * 1.08

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    title_font = load_font(34)
    metric_font = load_font(31)
    score_font = load_font(27)
    axis_font = load_font(23)

    draw_centered_text(
        draw,
        (0, 0, width, 32),
        "Top 3 drivers",
        title_font,
        (0, 0, 0, 255),
    )

    axis_left = 72
    axis_right = width - 72
    center_x = (axis_left + axis_right) / 2
    half_axis = (axis_right - axis_left) / 2

    draw_centered_text(
        draw,
        (axis_left - 66, 42, axis_left + 78, 76),
        "Female",
        axis_font,
        (0, 0, 0, 210),
    )
    draw_centered_text(
        draw,
        (axis_right - 78, 42, axis_right + 66, 76),
        "Male",
        axis_font,
        (0, 0, 0, 210),
    )

    for row_index, evidence in enumerate(evidence_rows):
        row_top = 78 + row_index * row_height
        axis_y = row_top + 48
        score = float(evidence["evidence"])
        metric = short_metric_label(str(evidence["metric"]))
        clipped_score = max(-limit, min(limit, score))
        point_x = center_x + (clipped_score / limit) * half_axis
        direction_color = (95, 174, 163, 255) if score > 0 else (214, 189, 88, 255)

        if len(metric) > 13:
            metric = metric[:10] + "..."
        draw.text((0, row_top), metric, font=metric_font, fill=(0, 0, 0, 255))
        score_text = f"{score:+.2f}"
        score_bbox = draw.textbbox((0, 0), score_text, font=score_font)
        draw.text(
            (width - (score_bbox[2] - score_bbox[0]), row_top + 1),
            score_text,
            font=score_font,
            fill=(0, 0, 0, 230),
        )

        draw.line((axis_left, axis_y, axis_right, axis_y), fill=(0, 0, 0, 145), width=4)
        draw.line((center_x, axis_y - 12, center_x, axis_y + 12), fill=(0, 0, 0, 180), width=3)
        draw.line((center_x, axis_y, point_x, axis_y), fill=direction_color, width=9)
        draw.ellipse(
            (point_x - 13, axis_y - 13, point_x + 13, axis_y + 13),
            fill=direction_color,
            outline=(0, 0, 0, 230),
            width=3,
        )

    return image


def make_graph_box(
    graph: Image.Image,
    padding: int,
    radius: int,
    bg_opacity: float,
    lollipop: Image.Image | None = None,
) -> Image.Image:
    padding = max(0, padding)
    radius = max(0, radius)
    bg_opacity = min(max(bg_opacity, 0.0), 1.0)
    gap = 16 if lollipop else 0
    content_width = graph.width + (gap + lollipop.width if lollipop else 0)
    content_height = max(graph.height, lollipop.height if lollipop else 0)

    box = Image.new(
        "RGBA",
        (
            content_width + 2 * padding,
            content_height + 2 * padding,
        ),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(box, "RGBA")
    rect = (0, 0, box.width - 1, box.height - 1)
    draw.rounded_rectangle(
        rect,
        radius=radius,
        fill=(255, 255, 255, round(255 * bg_opacity)),
    )
    graph_x = padding
    graph_y = padding + (content_height - graph.height) // 2
    box.alpha_composite(graph, (graph_x, graph_y))
    if lollipop:
        lollipop_x = padding + graph.width + gap
        lollipop_y = padding + (content_height - lollipop.height) // 2
        box.alpha_composite(lollipop, (lollipop_x, lollipop_y))
    return box


def draw_model_label(panel: Image.Image, label: str, index: int) -> None:
    draw = ImageDraw.Draw(panel, "RGBA")
    font = load_font(46)
    text = f"({chr(ord('a') + index)}) {label}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    pad_x = 18
    pad_y = 12
    x = 28
    y = 28
    rect = (x, y, x + text_width + 2 * pad_x, y + text_height + 2 * pad_y)
    draw.rounded_rectangle(rect, radius=8, fill=(255, 255, 255, 210))
    draw.text((x + pad_x, y + pad_y - bbox[1]), text, font=font, fill=(0, 0, 0, 255))


def compose_panel(
    base_path: Path,
    graph_path: Path,
    label: str,
    index: int,
    evidence: list[dict[str, str | float]] | None,
    overlay_scale: float,
    overlay_margin: int,
    graph_bg_opacity: float,
    graph_box_padding: int,
    graph_box_radius: int,
    lollipop_limit: float,
    draw_labels: bool,
) -> Image.Image:
    base = Image.open(base_path).convert("RGBA")
    graph = recolor_white_graph_text(trim_graph(Image.open(graph_path)))

    graph_width = round(base.width * overlay_scale)
    graph = resize_to_width(graph, graph_width)

    max_graph_height = round(base.height * 0.34)
    if graph.height > max_graph_height:
        graph = graph.resize(
            (round(graph.width * max_graph_height / graph.height), max_graph_height),
            Image.Resampling.LANCZOS,
        )

    lollipop = None
    if evidence:
        lollipop = make_lollipop_plot(
            evidence_rows=evidence,
            width=max(510, round(graph.width * 1.05)),
            limit=lollipop_limit,
        )

    graph_box = make_graph_box(
        graph=graph,
        padding=graph_box_padding,
        radius=graph_box_radius,
        bg_opacity=graph_bg_opacity,
        lollipop=lollipop,
    )
    x = round((base.width - graph_box.width) / 2)
    y = base.height - graph_box.height - overlay_margin
    base.alpha_composite(graph_box, (x, y))

    if draw_labels:
        draw_model_label(base, label, index)

    return base


def make_grid(
    panels: list[Image.Image],
    cols: int,
    gutter: int,
    outer_margin: int,
) -> Image.Image:
    if not panels:
        raise ValueError("Nenhum painel foi gerado.")

    panel_width = max(panel.width for panel in panels)
    panel_height = max(panel.height for panel in panels)
    rows = math.ceil(len(panels) / cols)

    canvas_width = cols * panel_width + (cols - 1) * gutter + 2 * outer_margin
    canvas_height = rows * panel_height + (rows - 1) * gutter + 2 * outer_margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    for index, panel in enumerate(panels):
        row, col = divmod(index, cols)
        x = outer_margin + col * (panel_width + gutter)
        y = outer_margin + row * (panel_height + gutter)
        canvas.paste(panel.convert("RGB"), (x, y))

    return canvas


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output.resolve()
    evidence_data = load_evidence_data(args)

    panels = []
    for index, model in enumerate(MODEL_ORDER):
        model_dir = input_dir / model
        if not model_dir.exists():
            raise FileNotFoundError(f"Pasta do modelo nao encontrada: {model_dir}")

        base_path = find_base_image(model_dir, model)
        graph_path = find_graph_image(model_dir)
        panel = compose_panel(
            base_path=base_path,
            graph_path=graph_path,
            label=model,
            index=index,
            evidence=evidence_data.get(model),
            overlay_scale=args.overlay_scale,
            overlay_margin=args.overlay_margin,
            graph_bg_opacity=args.graph_bg_opacity,
            graph_box_padding=args.graph_box_padding,
            graph_box_radius=args.graph_box_radius,
            lollipop_limit=args.lollipop_limit,
            draw_labels=not args.no_labels,
        )
        panels.append(panel)

    grid = make_grid(
        panels=panels,
        cols=2,
        gutter=args.gutter,
        outer_margin=args.outer_margin,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output, dpi=(args.dpi, args.dpi))

    if not args.no_pdf:
        pdf_output = output.with_suffix(".pdf")
        grid.save(pdf_output, "PDF", resolution=args.dpi)

    print(f"Quadro salvo em: {output}")
    if not args.no_pdf:
        print(f"PDF salvo em: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
