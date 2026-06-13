"""
@Dosya: config/schema.py
@Aciklama: LaflaAi-Core icin typed config sozlesmelerini ve preflight
            dogrulama hatalarini tanimlar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Bu modul OLMo tarzindaki typed config disiplinini Lafla icin temiz oda
        olarak yeniden ifade eder. Egitim kodu ham dict ile calismaz.
@Uyari: Buradaki varsayilanlar egitim davranisini etkiler. Yeni alanlar fail
        closed olacak sekilde dogrulanmalidir.
@Calisma-Semasi: load -> decode -> validate -> TrainingProfile
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


class ConfigError(ValueError):
    """Config dogrulama hatasi."""


def _require(condition: bool, message: str) -> None:
    """Kosul saglanmazsa ConfigError firlatir."""

    if not condition:
        raise ConfigError(message)


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Nested mapping alir ve tipini dogrular."""

    value = data.get(key)
    _require(isinstance(value, Mapping), f"{key} bolumu zorunlu")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    """String olmayan bir iterable'i string tuple'a cevirir."""

    _require(isinstance(value, Iterable) and not isinstance(value, (str, bytes)), f"{key} liste olmali")
    return tuple(str(item) for item in value)


def _int_tuple(value: Any, key: str) -> tuple[int, ...]:
    """String olmayan bir iterable'i int tuple'a cevirir."""

    _require(isinstance(value, Iterable) and not isinstance(value, (str, bytes)), f"{key} liste olmali")
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class RopeScalingConfig:
    """Uzun baglam icin RoPE konum olcekleme sozlesmesi."""

    type: str = "none"
    factor: float = 1.0
    original_context_length: int = 0

    @classmethod
    def from_value(cls, value: Any) -> "RopeScalingConfig":
        if value is None:
            return cls()
        _require(isinstance(value, Mapping), "rope_scaling mapping olmali")
        return cls(
            type=str(value.get("type", "none")),
            factor=float(value.get("factor", 1.0)),
            original_context_length=int(value.get("original_context_length", 0)),
        )

    def validate(self, context_length: int) -> None:
        _require(self.type in {"none", "linear", "dynamic"}, "desteklenmeyen rope_scaling type")
        _require(self.factor >= 1.0, "rope_scaling factor en az 1 olmali")
        if self.type == "none":
            _require(self.factor == 1.0, "rope_scaling none iken factor 1 olmali")
            return
        _require(self.original_context_length >= 512, "rope_scaling original_context_length cok kucuk")
        _require(
            context_length <= int(self.original_context_length * self.factor),
            "context_length rope_scaling kapasitesini asiyor",
        )


@dataclass(frozen=True)
class ModelConfig:
    """Decoder-only model hedefini tasir."""

    name: str
    family: str
    parameter_target: int
    vocab_size: int
    context_length: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    activation: str
    norm: str
    rope: bool
    qk_norm: bool
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = False
    use_bias: bool = False
    gradient_checkpointing: bool = True
    initializer_std: float = 0.02
    display_name: str = ""
    creator_name: str = ""
    identity_statement: str = ""
    attention_pattern: tuple[str, ...] = ("global",)
    sliding_window: int = 0
    rope_scaling: RopeScalingConfig = field(default_factory=RopeScalingConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelConfig":
        """Mapping verisinden ModelConfig uretir."""

        model = _mapping(data, "model")
        return cls(
            name=str(model["name"]),
            family=str(model["family"]),
            display_name=str(model.get("display_name", "")),
            creator_name=str(model.get("creator_name", "")),
            identity_statement=str(model.get("identity_statement", "")),
            parameter_target=int(model["parameter_target"]),
            vocab_size=int(model["vocab_size"]),
            context_length=int(model["context_length"]),
            hidden_size=int(model["hidden_size"]),
            intermediate_size=int(model["intermediate_size"]),
            num_layers=int(model["num_layers"]),
            num_attention_heads=int(model["num_attention_heads"]),
            num_key_value_heads=int(model["num_key_value_heads"]),
            activation=str(model["activation"]),
            norm=str(model["norm"]),
            rope=bool(model["rope"]),
            qk_norm=bool(model["qk_norm"]),
            rope_theta=float(model.get("rope_theta", 10000.0)),
            norm_eps=float(model.get("norm_eps", 1e-5)),
            dropout=float(model.get("dropout", 0.0)),
            tie_word_embeddings=bool(model.get("tie_word_embeddings", False)),
            use_bias=bool(model.get("use_bias", False)),
            gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
            initializer_std=float(model.get("initializer_std", 0.02)),
            attention_pattern=_string_tuple(model.get("attention_pattern", ("global",)), "attention_pattern"),
            sliding_window=int(model.get("sliding_window", 0)),
            rope_scaling=RopeScalingConfig.from_value(model.get("rope_scaling")),
        )

    def validate(self) -> None:
        """Model config alanlarini fail-closed dogrular."""

        _require(self.family == "decoder-only", "yalniz decoder-only desteklenir")
        for field_name, field_value in (
            ("display_name", self.display_name),
            ("creator_name", self.creator_name),
            ("identity_statement", self.identity_statement),
        ):
            if field_value:
                _require(field_value.strip() == field_value, f"{field_name} boslukla baslayamaz veya bitemez")
                _require("<|" not in field_value and "|>" not in field_value, f"{field_name} special token icermemeli")
        _require(100_000_000 <= self.parameter_target <= 1_500_000_000, "parametre hedefi 100M-1.5B araliginda olmali")
        _require(self.vocab_size % 128 == 0, "vocab_size 128'e bolunmeli")
        _require(self.context_length >= 512, "context_length cok kucuk")
        _require(self.hidden_size % self.num_attention_heads == 0, "hidden_size attention head'e bolunmeli")
        _require(self.num_attention_heads % self.num_key_value_heads == 0, "GQA icin head bolunmesi hatali")
        _require(self.activation in {"swiglu", "silu", "gelu"}, "desteklenmeyen activation")
        _require(self.norm in {"rmsnorm", "layernorm"}, "desteklenmeyen norm")
        _require(self.rope, "RoPE kapali olamaz")
        _require(self.rope_theta > 1.0, "rope_theta hatali")
        _require(bool(self.attention_pattern), "attention_pattern bos olamaz")
        _require(
            all(mode in {"local", "global"} for mode in self.attention_pattern),
            "attention_pattern yalniz local/global icerebilir",
        )
        if "local" in self.attention_pattern:
            _require(128 <= self.sliding_window <= self.context_length, "sliding_window guvenli aralik disinda")
        else:
            _require(self.sliding_window >= 0, "sliding_window negatif olamaz")
        self.rope_scaling.validate(self.context_length)
        _require(0.0 <= self.dropout < 0.5, "dropout guvenli aralik disinda")
        _require(0.0 < self.norm_eps <= 1e-3, "norm_eps guvenli aralik disinda")
        _require(0.0 < self.initializer_std <= 0.2, "initializer_std guvenli aralik disinda")

    def resolved_attention_pattern(self) -> tuple[str, ...]:
        """Kisa attention desenini tum katmanlara tekrar ederek acar."""

        return tuple(self.attention_pattern[index % len(self.attention_pattern)] for index in range(self.num_layers))


@dataclass(frozen=True)
class TrainingConfig:
    """Colab ve yerel egitim kosusu sozlesmesi."""

    max_steps: int
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    precision: str
    optimizer: str
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    warmup_steps: int
    save_every: int
    keep_last_checkpoints: int
    checkpoint_min_free_gb: float
    log_every: int
    seed: int
    require_drive_or_explicit_local_fallback: bool
    accelerator: str = "auto"
    sequence_curriculum: tuple[int, ...] = ()
    curriculum_token_boundaries: tuple[int, ...] = ()
    target_tokens: int = 0
    checkpoint_every_tokens: int = 0
    data_parallel: str = "off"
    distributed_backend: str = "auto"
    gradient_sync: str = "every_microstep"
    pin_memory: bool = False
    prefer_fused_optimizer: bool = False
    prefer_native_gqa: bool = False
    cuda_micro_batch_size_per_device: int = 0
    target_sequences_per_optimizer_step: int = 0
    gradient_checkpointing_min_sequence_length: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        """Mapping verisinden TrainingConfig uretir."""

        training = _mapping(data, "training")
        return cls(
            max_steps=int(training["max_steps"]),
            sequence_length=int(training["sequence_length"]),
            micro_batch_size=int(training["micro_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            precision=str(training["precision"]),
            optimizer=str(training["optimizer"]),
            learning_rate=float(training["learning_rate"]),
            min_learning_rate=float(training.get("min_learning_rate", 0.0)),
            weight_decay=float(training.get("weight_decay", 0.1)),
            grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
            warmup_steps=int(training["warmup_steps"]),
            save_every=int(training["save_every"]),
            keep_last_checkpoints=int(training["keep_last_checkpoints"]),
            checkpoint_min_free_gb=float(training.get("checkpoint_min_free_gb", 2.0)),
            log_every=int(training.get("log_every", 10)),
            seed=int(training.get("seed", 1337)),
            require_drive_or_explicit_local_fallback=bool(training["require_drive_or_explicit_local_fallback"]),
            accelerator=str(training.get("accelerator", "auto")),
            sequence_curriculum=_int_tuple(training.get("sequence_curriculum", ()), "sequence_curriculum"),
            curriculum_token_boundaries=_int_tuple(
                training.get("curriculum_token_boundaries", ()),
                "curriculum_token_boundaries",
            ),
            target_tokens=int(training.get("target_tokens", 0)),
            checkpoint_every_tokens=int(training.get("checkpoint_every_tokens", 0)),
            data_parallel=str(training.get("data_parallel", "off")),
            distributed_backend=str(training.get("distributed_backend", "auto")),
            gradient_sync=str(training.get("gradient_sync", "every_microstep")),
            pin_memory=bool(training.get("pin_memory", False)),
            prefer_fused_optimizer=bool(training.get("prefer_fused_optimizer", False)),
            prefer_native_gqa=bool(training.get("prefer_native_gqa", False)),
            cuda_micro_batch_size_per_device=int(training.get("cuda_micro_batch_size_per_device", 0)),
            target_sequences_per_optimizer_step=int(training.get("target_sequences_per_optimizer_step", 0)),
            gradient_checkpointing_min_sequence_length=int(
                training.get("gradient_checkpointing_min_sequence_length", 0)
            ),
        )

    def validate(self) -> None:
        """Training config alanlarini dogrular."""

        _require(self.max_steps > 0, "max_steps pozitif olmali")
        _require(self.sequence_length >= 128, "sequence_length cok kucuk")
        _require(self.micro_batch_size >= 1, "micro_batch_size pozitif olmali")
        _require(self.gradient_accumulation_steps >= 1, "gradient_accumulation_steps pozitif olmali")
        _require(self.precision in {"fp16", "bf16", "fp32"}, "desteklenmeyen precision")
        _require(self.optimizer in {"adamw", "adamw8bit", "lion"}, "desteklenmeyen optimizer")
        _require(self.accelerator in {"auto", "cuda", "xla", "cpu"}, "desteklenmeyen accelerator")
        _require(self.data_parallel in {"off", "auto", "on"}, "desteklenmeyen data_parallel")
        _require(
            self.distributed_backend in {"auto", "nccl", "gloo"},
            "desteklenmeyen distributed_backend",
        )
        _require(
            self.gradient_sync in {"every_microstep", "final_microstep"},
            "desteklenmeyen gradient_sync",
        )
        if self.accelerator == "xla":
            _require(self.optimizer == "adamw", "TPU/XLA icin optimizer=adamw olmali")
            _require(self.precision in {"bf16", "fp32"}, "TPU/XLA icin bf16 veya fp32 kullanilmali")
            _require(self.data_parallel != "on", "data_parallel=on XLA/TPU icin kullanilamaz")
            _require(self.distributed_backend == "auto", "XLA/TPU distributed_backend auto olmali")
        if self.accelerator == "cpu":
            _require(self.data_parallel != "on", "data_parallel=on CPU icin kullanilamaz")
            _require(self.distributed_backend != "nccl", "CPU icin NCCL kullanilamaz")
        _require(0.0 < self.learning_rate <= 0.01, "learning_rate guvenli aralik disinda")
        _require(0.0 <= self.min_learning_rate <= self.learning_rate, "min_learning_rate hatali")
        _require(0.0 <= self.weight_decay <= 1.0, "weight_decay guvenli aralik disinda")
        _require(0.0 < self.grad_clip_norm <= 100.0, "grad_clip_norm guvenli aralik disinda")
        _require(0 <= self.warmup_steps < self.max_steps, "warmup_steps hatali")
        _require(self.save_every > 0, "save_every pozitif olmali")
        _require(self.keep_last_checkpoints >= 1, "en az bir checkpoint tutulmali")
        _require(self.checkpoint_min_free_gb >= 0.0, "checkpoint_min_free_gb negatif olamaz")
        _require(self.log_every > 0, "log_every pozitif olmali")
        _require(0 <= self.seed <= 2**32 - 1, "seed guvenli aralik disinda")
        if self.sequence_curriculum or self.curriculum_token_boundaries:
            _require(
                len(self.sequence_curriculum) == len(self.curriculum_token_boundaries),
                "sequence_curriculum ve curriculum_token_boundaries ayni uzunlukta olmali",
            )
            _require(self.curriculum_token_boundaries[0] == 0, "ilk curriculum token siniri 0 olmali")
            _require(
                all(left < right for left, right in zip(self.curriculum_token_boundaries, self.curriculum_token_boundaries[1:])),
                "curriculum_token_boundaries artan olmali",
            )
            _require(
                all(length >= 128 for length in self.sequence_curriculum),
                "sequence_curriculum uzunluklari en az 128 olmali",
            )
            _require(
                all(left < right for left, right in zip(self.sequence_curriculum, self.sequence_curriculum[1:])),
                "sequence_curriculum artan olmali",
            )
            _require(self.target_tokens > 0, "curriculum icin target_tokens pozitif olmali")
            _require(
                self.curriculum_token_boundaries[-1] < self.target_tokens,
                "son curriculum siniri target_tokens degerinden kucuk olmali",
            )
        _require(self.target_tokens >= 0, "target_tokens negatif olamaz")
        _require(self.checkpoint_every_tokens >= 0, "checkpoint_every_tokens negatif olamaz")
        _require(self.cuda_micro_batch_size_per_device >= 0, "cuda_micro_batch_size_per_device negatif olamaz")
        _require(
            self.target_sequences_per_optimizer_step >= 0,
            "target_sequences_per_optimizer_step negatif olamaz",
        )
        _require(
            (self.cuda_micro_batch_size_per_device == 0) == (self.target_sequences_per_optimizer_step == 0),
            "cuda batch tuning alanlari birlikte etkinlestirilmeli",
        )
        _require(
            self.gradient_checkpointing_min_sequence_length >= 0,
            "gradient_checkpointing_min_sequence_length negatif olamaz",
        )


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer hedefini ve kalite kapilarini tasir."""

    kind: str
    vocab_size: int
    normalization: str
    required_special_tokens: tuple[str, ...]
    quality_gates: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TokenizerConfig":
        """Mapping verisinden TokenizerConfig uretir."""

        tokenizer = _mapping(data, "tokenizer")
        tokens = tokenizer.get("required_special_tokens", ())
        _require(isinstance(tokens, Iterable), "required_special_tokens liste olmali")
        gates = tokenizer.get("quality_gates", {})
        _require(isinstance(gates, Mapping), "quality_gates mapping olmali")
        return cls(
            kind=str(tokenizer["kind"]),
            vocab_size=int(tokenizer["vocab_size"]),
            normalization=str(tokenizer["normalization"]),
            required_special_tokens=tuple(str(token) for token in tokens),
            quality_gates=gates,
        )

    def validate(self) -> None:
        """Tokenizer config alanlarini dogrular."""

        _require(self.kind == "bpe", "ilk hedef yalniz BPE")
        _require(self.vocab_size % 128 == 0, "tokenizer vocab_size 128'e bolunmeli")
        _require(self.normalization == "utf8_nfc", "UTF-8 NFC zorunlu")
        required = {"<|bos|>", "<|eos|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|think|>", "<|/think|>"}
        missing = required.difference(self.required_special_tokens)
        _require(not missing, f"eksik special token: {sorted(missing)}")


@dataclass(frozen=True)
class RuntimeConfig:
    """Dusuk guclu calistirma hedefini tasir."""

    target: str
    quantization: str
    context_length: int
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    memory_budget_gb: float
    thinking_effort: str = "medium"
    thinking_budget_tokens: int = 192
    developer_mode: bool = False
    raw_thinking_visible: bool = False
    safety_profile: str = "standard"
    context_overflow_strategy: str = "sliding_window"
    system_prompt_leak_guard: bool = True
    turkish_quality_guard: bool = True
    cache_implementation: str = "dynamic"
    cache_dtype: str = "bf16"
    weight_dtype: str = "fp16"
    batch_size: int = 1
    summary_trigger_tokens: int = 0
    summary_max_tokens: int = 0
    preserve_recent_tokens: int = 0
    retrieval_max_tokens: int = 0
    output_reserve_tokens: int = 0
    peak_rss_limit_mib: int = 0
    max_concurrent_generations: int = 1
    runtime_overhead_mib: int = 256
    allocator_headroom_ratio: float = 0.15

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RuntimeConfig":
        """Mapping verisinden RuntimeConfig uretir."""

        runtime = _mapping(data, "runtime")
        return cls(
            target=str(runtime["target"]),
            quantization=str(runtime["quantization"]),
            context_length=int(runtime["context_length"]),
            max_new_tokens=int(runtime["max_new_tokens"]),
            temperature=float(runtime["temperature"]),
            top_p=float(runtime["top_p"]),
            repetition_penalty=float(runtime["repetition_penalty"]),
            memory_budget_gb=float(runtime["memory_budget_gb"]),
            thinking_effort=str(runtime.get("thinking_effort", "medium")),
            thinking_budget_tokens=int(runtime.get("thinking_budget_tokens", 192)),
            developer_mode=bool(runtime.get("developer_mode", False)),
            raw_thinking_visible=bool(runtime.get("raw_thinking_visible", False)),
            safety_profile=str(runtime.get("safety_profile", "standard")),
            context_overflow_strategy=str(runtime.get("context_overflow_strategy", "sliding_window")),
            system_prompt_leak_guard=bool(runtime.get("system_prompt_leak_guard", True)),
            turkish_quality_guard=bool(runtime.get("turkish_quality_guard", True)),
            cache_implementation=str(runtime.get("cache_implementation", "dynamic")),
            cache_dtype=str(runtime.get("cache_dtype", "bf16")),
            weight_dtype=str(runtime.get("weight_dtype", "fp16")),
            batch_size=int(runtime.get("batch_size", 1)),
            summary_trigger_tokens=int(runtime.get("summary_trigger_tokens", 0)),
            summary_max_tokens=int(runtime.get("summary_max_tokens", 0)),
            preserve_recent_tokens=int(runtime.get("preserve_recent_tokens", 0)),
            retrieval_max_tokens=int(runtime.get("retrieval_max_tokens", 0)),
            output_reserve_tokens=int(runtime.get("output_reserve_tokens", 0)),
            peak_rss_limit_mib=int(runtime.get("peak_rss_limit_mib", 0)),
            max_concurrent_generations=int(runtime.get("max_concurrent_generations", 1)),
            runtime_overhead_mib=int(runtime.get("runtime_overhead_mib", 256)),
            allocator_headroom_ratio=float(runtime.get("allocator_headroom_ratio", 0.15)),
        )

    def validate(self) -> None:
        """Runtime config alanlarini dogrular."""

        _require(self.target in {"desktop-cpu", "mobile-lab", "research-gpu"}, "desteklenmeyen runtime target")
        _require(self.quantization in {"none", "8bit", "4bit"}, "desteklenmeyen quantization")
        _require(self.context_length >= 256, "runtime context cok kucuk")
        _require(1 <= self.max_new_tokens <= 2048, "max_new_tokens guvenli aralik disinda")
        _require(0.0 < self.temperature <= 2.0, "temperature guvenli aralik disinda")
        _require(0.0 < self.top_p <= 1.0, "top_p guvenli aralik disinda")
        _require(self.memory_budget_gb > 0, "memory_budget_gb pozitif olmali")
        _require(self.thinking_effort in {"low", "medium", "high"}, "desteklenmeyen thinking_effort")
        _require(0 <= self.thinking_budget_tokens <= 4096, "thinking_budget_tokens guvenli aralik disinda")
        _require(self.safety_profile in {"standard", "research"}, "desteklenmeyen safety_profile")
        _require(
            self.context_overflow_strategy in {"truncate_oldest", "sliding_window", "summarize_then_slide"},
            "desteklenmeyen context_overflow_strategy",
        )
        _require(self.cache_implementation in {"dynamic", "static", "quantized"}, "desteklenmeyen cache_implementation")
        _require(self.cache_dtype in {"fp32", "fp16", "bf16", "int8"}, "desteklenmeyen cache_dtype")
        _require(self.weight_dtype in {"fp32", "fp16", "bf16", "int8", "int4"}, "desteklenmeyen weight_dtype")
        _require(self.batch_size >= 1, "batch_size pozitif olmali")
        _require(self.max_concurrent_generations >= 1, "max_concurrent_generations pozitif olmali")
        _require(self.summary_trigger_tokens >= 0, "summary_trigger_tokens negatif olamaz")
        _require(self.summary_max_tokens >= 0, "summary_max_tokens negatif olamaz")
        _require(self.preserve_recent_tokens >= 0, "preserve_recent_tokens negatif olamaz")
        _require(self.retrieval_max_tokens >= 0, "retrieval_max_tokens negatif olamaz")
        _require(self.output_reserve_tokens >= 0, "output_reserve_tokens negatif olamaz")
        _require(self.peak_rss_limit_mib >= 0, "peak_rss_limit_mib negatif olamaz")
        _require(self.runtime_overhead_mib >= 0, "runtime_overhead_mib negatif olamaz")
        _require(0.0 <= self.allocator_headroom_ratio <= 1.0, "allocator_headroom_ratio 0-1 araliginda olmali")
        if self.summary_trigger_tokens:
            _require(
                self.context_overflow_strategy == "summarize_then_slide",
                "summary trigger icin summarize_then_slide zorunlu",
            )
            _require(self.summary_trigger_tokens < self.context_length, "summary trigger context sinirindan kucuk olmali")
            _require(self.summary_max_tokens > 0, "summary_max_tokens pozitif olmali")
            _require(self.preserve_recent_tokens > 0, "preserve_recent_tokens pozitif olmali")
        reserved = (
            self.summary_max_tokens
            + self.preserve_recent_tokens
            + self.retrieval_max_tokens
            + self.output_reserve_tokens
        )
        _require(reserved <= self.context_length, "runtime ayrilmis token butceleri context_length degerini asiyor")
        if self.output_reserve_tokens:
            _require(self.output_reserve_tokens >= self.max_new_tokens, "output_reserve_tokens max_new_tokens altinda olamaz")
        if self.raw_thinking_visible:
            _require(self.developer_mode, "raw_thinking_visible icin developer_mode zorunlu")
            _require(self.safety_profile == "research", "raw_thinking_visible yalniz research profilinde acilir")
        if self.safety_profile == "research":
            _require(self.developer_mode, "research safety_profile icin developer_mode zorunlu")


@dataclass(frozen=True)
class PostTrainingConfig:
    """SFT/DPO/thinking uyum asamasi sozlesmesini tasir."""

    stage: str
    sequence_length: int
    learning_rate: float
    epochs: int
    label_policy: str
    only_last_assistant: bool
    public_thinking_visible: bool
    max_thinking_chars: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PostTrainingConfig":
        """Mapping verisinden PostTrainingConfig uretir."""

        post_training = _mapping(data, "post_training")
        return cls(
            stage=str(post_training["stage"]),
            sequence_length=int(post_training["sequence_length"]),
            learning_rate=float(post_training["learning_rate"]),
            epochs=int(post_training["epochs"]),
            label_policy=str(post_training["label_policy"]),
            only_last_assistant=bool(post_training["only_last_assistant"]),
            public_thinking_visible=bool(post_training["public_thinking_visible"]),
            max_thinking_chars=int(post_training["max_thinking_chars"]),
        )

    def validate(self) -> None:
        """Post-training config alanlarini dogrular."""

        _require(self.stage in {"thinking_sft", "instruction_sft", "dpo"}, "desteklenmeyen post_training stage")
        _require(128 <= self.sequence_length <= 32768, "post_training sequence_length guvenli aralik disinda")
        _require(0.0 < self.learning_rate <= 0.001, "post_training learning_rate guvenli aralik disinda")
        _require(1 <= self.epochs <= 5, "post_training epochs guvenli aralik disinda")
        _require(
            self.label_policy in {"assistant_only", "assistant_with_thinking", "last_assistant_only"},
            "desteklenmeyen label_policy",
        )
        _require(not self.public_thinking_visible, "public_thinking_visible true olamaz")
        _require(1 <= self.max_thinking_chars <= 16000, "max_thinking_chars guvenli aralik disinda")
