"""
 _        _    _____ _        _
| |      / \\  |  ___| |      / \\
| |     / _ \\ | |_  | |     / _ \\
| |___ / ___ \\|  _| | |___ / ___ \\
|_____/_/   \\_\\_|   |_____/_/   \\_\\

@Dosya: lafla_1b_sifirdan_egitim.py
@Açıklama: Colab üzerinde Lafla AI modelini sıfırdan eğitmek için giriş scripti.
@Yazar: Lafla Geliştirme Ekibi
@Bilgi: Smoke modu yalnızca modelin çalıştığını doğrular; üretim eğitimi gerçek JSONL ve tokenizer ister.
@Uyarı: Hazır GPT, Llama, Mistral, Qwen veya Claude ağırlığı yüklemez.
@Çalışma-Şeması: config -> identity -> dataset/smoke batch -> train loop -> checkpoint
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lafla_dataset import PackedJsonlDataset
from lafla_model import LaflaModelConfig, LaflaTransformer, estimate_parameters
from lafla_persona import build_system_prompt, load_identity


def main() -> None:
    """Eğitim yapılandırmasını okuyup smoke veya gerçek veri eğitimini başlatır."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-config", default=str(ROOT / "konfigurasyon" / "lafla-1b-model.json"))
    parser.add_argument("--train-config", default=str(ROOT / "konfigurasyon" / "egitim-colab-smoke.json"))
    parser.add_argument("--identity-config", default=str(ROOT / "konfigurasyon" / "lafla-ai-kimlik.json"))
    parser.add_argument("--data-jsonl", action="append", help="İşlenmiş JSONL shard yolu. Smoke dışında zorunludur.")
    parser.add_argument("--tokenizer-path", help="lafla_tokenizer.py ile eğitilmiş tokenizer JSON yolu. Smoke dışında zorunludur.")
    args = parser.parse_args()

    train_config = read_json(args.train_config)
    identity = load_identity(args.identity_config)
    random.seed(train_config["seed"])
    torch.manual_seed(train_config["seed"])

    if args.smoke:
        model_config = smoke_model_config(train_config)
        dataloader = None
    else:
        require_training_inputs(args.data_jsonl, args.tokenizer_path)
        model_config = LaflaModelConfig(**read_json(args.model_config))
        dataset = PackedJsonlDataset(
            [Path(item) for item in args.data_jsonl or []],
            Path(args.tokenizer_path or ""),
            train_config["sequence_length"],
        )
        dataloader = cycle_loader(dataset, train_config["batch_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LaflaTransformer(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])

    print(json.dumps({
        "device": device,
        "identity": identity.name,
        "parameters": estimate_parameters(model_config),
        "smoke": args.smoke,
        "system_prompt_preview": build_system_prompt(identity).split("\n")[0],
    }, ensure_ascii=False))

    train_loop(model, optimizer, model_config, train_config, dataloader, device, args.smoke)
    save_checkpoint(model, model_config, train_config, args.smoke)


def train_loop(
    model: LaflaTransformer,
    optimizer: AdamW,
    model_config: LaflaModelConfig,
    train_config: dict,
    dataloader: object,
    device: str,
    smoke: bool,
) -> None:
    """Gradient accumulation ve checkpoint öncesi ana eğitim döngüsünü çalıştırır."""

    model.train()
    for step in range(train_config["max_steps"]):
        loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(train_config["gradient_accumulation_steps"]):
            batch = next_batch(dataloader, train_config, model_config, device, smoke)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda" and train_config["mixed_precision"]):
                out = model(batch, labels=batch)
                loss = out["loss"] / train_config["gradient_accumulation_steps"]
            loss.backward()
            loss_accum += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 5 == 0:
            print(json.dumps({"step": step, "loss": loss_accum}, ensure_ascii=False))


def next_batch(dataloader: object, train_config: dict, model_config: LaflaModelConfig, device: str, smoke: bool) -> torch.Tensor:
    """Smoke modunda sentetik, gerçek modda paketlenmiş veri batch'i döndürür."""

    if smoke:
        return torch.randint(0, model_config.vocab_size, (train_config["batch_size"], train_config["sequence_length"]), dtype=torch.long, device=device)
    batch = next(dataloader)
    return batch.to(device)


def cycle_loader(dataset: PackedJsonlDataset, batch_size: int):
    """DataLoader bittikçe başa sararak sabit eğitim akışı üretir."""

    while True:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for batch in loader:
            yield batch


def save_checkpoint(model: LaflaTransformer, model_config: LaflaModelConfig, train_config: dict, smoke: bool) -> None:
    """Model ağırlıklarını ve config özetini checkpoint dizinine yazar."""

    checkpoint_dir = Path(train_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name = "lafla-smoke.pt" if smoke else "lafla-train-step-final.pt"
    path = checkpoint_dir / name
    torch.save({"model": model.state_dict(), "config": model_config.__dict__}, path)
    print(json.dumps({"checkpoint": str(path)}, ensure_ascii=False))


def smoke_model_config(train_config: dict) -> LaflaModelConfig:
    """Colab veya yerel ortamda hızlı doğrulama için küçük model config'i üretir."""

    return LaflaModelConfig(
        vocab_size=4096,
        context_length=train_config["sequence_length"],
        hidden_size=256,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=768,
        dropout=0.0,
        tie_word_embeddings=False,
        use_bias=False,
    )


def require_training_inputs(data_jsonl: list[str] | None, tokenizer_path: str | None) -> None:
    """Smoke dışındaki eğitimde veri ve tokenizer yoksa kapalı şekilde hata verir."""

    if not data_jsonl:
        raise ValueError("--data-jsonl is required outside --smoke")
    if not tokenizer_path:
        raise ValueError("--tokenizer-path is required outside --smoke")


def read_json(path: str) -> dict:
    """UTF-8 JSON dosyasını sözlük olarak okur."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
