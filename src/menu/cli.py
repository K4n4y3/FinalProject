"""Command-line interface: ``menu --help``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, help="Menu from Neural Networks — data, model bundle and service.")


@app.command("build-data")
def build_data(
    raw_json: Annotated[Path, typer.Argument(help="full_format_recipes.json (Kaggle hugodarwood/epirecipes)")],
    out: Annotated[Path, typer.Option(help="output CSV")] = Path("data/epicurious_clean_split.csv"),
) -> None:
    """Re-run the Assignment 2 cleaning + grouped split from the raw Epicurious JSON."""
    from menu.data import build_split, check_no_leakage

    df, log = build_split(raw_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["ingredients", "directions", "categories"], errors="ignore").to_csv(out, index=False)
    typer.echo(
        json.dumps(
            {"rows": df.split.value_counts().to_dict(), "leakage": check_no_leakage(df), "steps": log.steps},
            indent=2,
            default=str,
        )
    )


@app.command("build-bundle")
def build_bundle_cmd(
    data: Annotated[Path, typer.Option(help="frozen split CSV")] = Path("data/epicurious_clean_split.csv"),
    out: Annotated[Path, typer.Option(help="bundle directory")] = Path("artifacts"),
    version: Annotated[str, typer.Option(help="bundle version tag")] = "1.0.0",
    lora: Annotated[Path | None, typer.Option(help="A4 LoRA adapter dir")] = None,
    lora_rag: Annotated[Path | None, typer.Option(help="LoRA-RAG adapter dir")] = None,
    eval_report: Annotated[Path | None, typer.Option(help="final_eval.json from the training notebook")] = None,
    dense: Annotated[bool, typer.Option(help="embed the KB with all-MiniLM-L6-v2 (needs the llm extra)")] = False,
) -> None:
    """Train the classifier and the retriever, copy adapters and write a checksummed manifest."""
    from menu.bundle import build_bundle

    adapters: dict[str, str | Path] = {k: v for k, v in {"lora": lora, "lora_rag": lora_rag}.items() if v is not None}
    report = json.loads(eval_report.read_text()) if eval_report else {}
    generator = {"final_variant": report["final_variant"]} if "final_variant" in report else None
    if generator is None and "lora" not in adapters and "lora_rag" in adapters:
        generator = {
            "final_variant": {
                "name": "Qwen + LoRA-RAG",
                "adapter": "lora_rag",
                "rag_k": 3,
                "decoding": "beam4",
            }
        }
    encoder = None
    if dense:
        from menu.rag.retriever import sentence_encoder

        encoder = sentence_encoder()
    manifest = build_bundle(
        data,
        out,
        version,
        adapters=adapters,
        generator=generator,
        generator_eval=report.get("evaluation"),
        encoder=encoder,
    )
    typer.echo(
        json.dumps({k: manifest[k] for k in ("bundle_version", "classifier", "generator")}, indent=2, default=str)[
            :3000
        ]
    )


@app.command()
def verify(root: Path = Path("artifacts")) -> None:
    """Check every file of the bundle against the SHA-256 in its manifest."""
    from menu.bundle import verify_bundle

    bad = verify_bundle(root)
    if bad:
        typer.echo(f"MISMATCH: {bad}", err=True)
        raise typer.Exit(1)
    typer.echo("bundle OK")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 7860, workers: int = 1) -> None:
    """Run the API + UI (uvicorn)."""
    import uvicorn

    uvicorn.run("menu.api.main:app", host=host, port=port, workers=workers, proxy_headers=True)


@app.command("eval-retriever")
def eval_retriever(data: Path = Path("data/epicurious_clean_split.csv"), k: int = 5) -> None:
    """Retrieval quality on validation: title ROUGE of neighbours and dish-type agreement."""
    import numpy as np

    from menu.data import load_split
    from menu.metrics import rouge_l
    from menu.rag.retriever import RecipeRetriever

    df = load_split(data)
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    hits = RecipeRetriever().fit(tr).search_batch(va.ing_clean.tolist(), k=k, mode="sparse")
    lab = va.course.notna().to_numpy()
    res = {
        "rougeL@1": float(np.mean([rouge_l(h[0].title, t) if h else 0 for h, t in zip(hits, va.title, strict=True)])),
        f"best_rougeL@{k}": float(
            np.mean([max([rouge_l(x.title, t) for x in h] or [0]) for h, t in zip(hits, va.title, strict=True)])
        ),
        "course_agreement@3": float(
            np.mean(
                [np.mean([x.course == c for x in h[:3]]) for h, c, ok in zip(hits, va.course, lab, strict=True) if ok]
            )
        ),
    }
    typer.echo(json.dumps(res, indent=2))


if __name__ == "__main__":
    app()
