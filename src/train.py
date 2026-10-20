"""
Stage 4: Training Loop with Cross-Validation

Implements:
    1. GAT training with multi-task loss (main + auxiliary)
    2. 5-fold stratified cross-validation
    3. Hybrid GAT -> Calibrated Random Forest pipeline
    4. Early stopping and learning rate scheduling

References:
    - Multi-task loss: Rahaman et al., 2023 (main + 0.3 * aux_loss)
    - CV strategy: 5-fold stratified (Alharbi et al., 2025)
    - Optimizer: Adam, lr=0.001, weight_decay=1e-4 (Alharbi et al., 2025)
    - Scheduler: ReduceLROnPlateau (Alharbi et al., 2025)
    - Hybrid GNN+RF: Palmal et al., 2024
"""

import os
import copy
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from collections import defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from lifelines.utils import concordance_index

from src.model import build_model, HybridModel
from src.dataset import (
    BreastCancerGraphDataset,
    get_dataloaders,
    build_dataset,
    compute_class_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def deephit_loss(
    logits: torch.Tensor,
    bin_idx: torch.Tensor,
    censored: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """DeepHit-style discrete-time survival loss (Lee et al., 2018).

    Mass vs. ranking trade-off: this implements just the mass term (L_1 in
    the paper). The ranking term is handled separately by the Cox head.

    Semantics per-patient:
      - Uncensored (event observed at bin k): maximize P(event=k).
      - Censored (last-seen at bin k): maximize P(survive past k)
        = sum_{j>k} P(event=j), i.e. the patient survived all bins up to k.

    Args:
        logits: [B, K] unnormalized class scores (main classification head).
        bin_idx: [B] int, which survival bin the patient is in.
        censored: [B] float 1.0 if censored, 0.0 if event observed.

    Returns:
        Scalar loss averaged over the batch. Returns 0 when the batch has
        no valid targets (bin_idx < 0).
    """
    valid = bin_idx >= 0
    if valid.sum() < 1:
        return torch.zeros((), device=logits.device)

    logits = logits[valid]
    bin_idx = bin_idx[valid].long()
    censored = censored[valid].float()

    probs = torch.softmax(logits, dim=-1).clamp(min=eps, max=1.0 - eps)  # [B, K]

    # P(event at bin_idx) -- uncensored signal
    event_p = probs.gather(1, bin_idx.unsqueeze(1)).squeeze(1)  # [B]
    event_ll = -torch.log(event_p + eps)

    # P(survive > bin_idx) = 1 - cumsum_up_to_bin_idx
    cum = probs.cumsum(dim=-1)  # [B, K]
    surv_p = 1.0 - cum.gather(1, bin_idx.unsqueeze(1)).squeeze(1)
    surv_p = surv_p.clamp(min=eps)
    surv_ll = -torch.log(surv_p)

    # Combine: censored -> surv_ll, uncensored -> event_ll
    loss = (1.0 - censored) * event_ll + censored * surv_ll
    return loss.mean()


def cox_ph_loss(log_hazards: torch.Tensor, times: torch.Tensor, events: torch.Tensor,
                eps: float = 1e-7) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow approximation).

    Reference: Katzman et al., 2018 -- DeepSurv.

    Args:
        log_hazards: [N] predicted scalar log-hazard per patient.
        times: [N] observed event/censoring times (>0).
        events: [N] binary indicator -- 1 if event observed, 0 if censored.

    Returns:
        Scalar loss averaged over observed events. Returns 0 when there are
        no events in the batch (gradient still valid because the subtraction
        short-circuits).
    """
    # Drop invalid rows (e.g. SMOTE-synthetic samples with NaN times)
    valid = (~torch.isnan(times)) & (~torch.isnan(events.float())) & (times > 0)
    if valid.sum() < 2:
        return torch.zeros((), device=log_hazards.device)

    h = log_hazards[valid]
    t = times[valid]
    e = events[valid].float()

    if e.sum() < 1:
        return torch.zeros((), device=log_hazards.device)

    # Sort by time descending so prefix cumsum gives the risk set
    _, sort_idx = torch.sort(t, descending=True)
    h = h[sort_idx]
    e = e[sort_idx]

    # Numerically stable cumulative logsumexp
    hmax = h.max().detach()
    log_cum = torch.log(torch.cumsum(torch.exp(h - hmax), dim=0) + eps) + hmax

    pll = (h - log_cum) * e
    n_events = e.sum().clamp(min=1.0)
    return -(pll.sum() / n_events)


LOSS_NAMES = ("ce", "deephit", "cox", "ordinal", "aux_stage")


def _compute_loss_terms(
    batch,
    cls_logits, aux_out, cox_out, ordinal_out, bin_logits,
    main_criterion, aux_criterion, ordinal_criterion,
    active_losses,
    device,
) -> dict:
    """Compute every loss term from ONE forward pass.

    Inactive terms are `.detach()`ed -- they appear in the returned dict for
    logging purposes but cannot propagate a gradient. The caller is expected
    to sum only `k in active_losses` when building the optimizer target.

    For survival terms that require non-NaN event info, we return a
    device-local zero (already detached via factory) when the batch has no
    valid rows -- the loss is effectively masked.
    """
    terms = {}

    # CE is always computable (y is never NaN post-SMOTE).
    ce = main_criterion(cls_logits, batch.y)
    terms["ce"] = ce if "ce" in active_losses else ce.detach()

    # DeepHit: needs os_event. SMOTE-synthetic rows have NaN os_event and
    # are masked out so they don't corrupt the censoring indicator.
    dh = torch.zeros((), device=device)
    if hasattr(batch, "os_event") and batch.os_event is not None:
        event = batch.os_event.float()
        valid = ~torch.isnan(event)
        if valid.any():
            dh = deephit_loss(
                bin_logits[valid], batch.y[valid], 1.0 - event[valid]
            )
    terms["deephit"] = dh if "deephit" in active_losses else dh.detach()

    # Cox partial likelihood.
    cox = torch.zeros((), device=device)
    if hasattr(batch, "os_time") and batch.os_time is not None:
        cox = cox_ph_loss(cox_out, batch.os_time.float(), batch.os_event.float())
    terms["cox"] = cox if "cox" in active_losses else cox.detach()

    # Ordinal: SmoothL1 on float class index. Always computable.
    ord_loss = ordinal_criterion(ordinal_out, batch.y.float())
    terms["ordinal"] = ord_loss if "ordinal" in active_losses else ord_loss.detach()

    # Aux stage: needs parseable tumor_stage (>=0 after _parse_stage).
    aux = torch.zeros((), device=device)
    if hasattr(batch, "tumor_stage") and batch.tumor_stage is not None:
        valid = batch.tumor_stage >= 0
        if valid.any():
            aux = aux_criterion(aux_out[valid], batch.tumor_stage[valid])
    terms["aux_stage"] = aux if "aux_stage" in active_losses else aux.detach()

    return terms


def train_one_epoch(
    model: HybridModel,
    loader,
    optimizer,
    device,
    active_losses: list = None,
    loss_weights: dict = None,
    label_smoothing: float = 0.1,
    class_weights: torch.Tensor = None,
) -> dict:
    """Train for one epoch with a configurable subset of the multi-task loss.

    `active_losses` is a list/set subset of LOSS_NAMES. Only these terms
    contribute to `loss.backward()`; the rest are logged (detached) for
    reporting. This supports the Phase-1.5 loss-ablation sweep (see
    tools/run_ablation_sweep.py).

    `loss_weights` maps loss-name -> scalar weight; defaults to 1.0.

    Head decoupling (Phase 1 fix): the main classifier head (self.fc) is
    supervised purely by class-weighted smoothed CE. DeepHit supervises its
    own head (self.survival_bin_head).
    """
    if active_losses is None:
        active_losses = list(LOSS_NAMES)
    active_set = set(active_losses)
    if loss_weights is None:
        loss_weights = {}

    model.train()
    sums = {k: 0.0 for k in LOSS_NAMES}
    total_loss = 0.0
    correct = 0
    total = 0

    main_criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )
    aux_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    ordinal_criterion = nn.SmoothL1Loss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        gene_idx = getattr(batch, "gene_idx", None)
        cls_logits, aux_out, _, cox_out, ordinal_out, bin_logits = model(
            batch.x, batch.edge_index, batch.batch, batch.clinical,
            gene_idx=gene_idx,
        )

        terms = _compute_loss_terms(
            batch, cls_logits, aux_out, cox_out, ordinal_out, bin_logits,
            main_criterion, aux_criterion, ordinal_criterion,
            active_set, device,
        )

        # Build optimizer target from active terms ONLY. Skipping (rather
        # than zero-weighting) inactive terms is deliberate: backward through
        # a 0*tensor still traces the graph and can introduce small
        # numerical disturbances. With `.detach()` on inactive terms, PyTorch
        # treats them as constants and the summation below ignores them.
        active_grad_terms = {
            k: terms[k] for k in active_set if terms[k].requires_grad
        }
        if active_grad_terms:
            loss = sum(
                loss_weights.get(k, 1.0) * v for k, v in active_grad_terms.items()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch.num_graphs
        # else: nothing to backprop this batch (e.g. deephit-only run on a
        # batch where every os_event is NaN). Skip the step.

        bs = batch.num_graphs
        for k in LOSS_NAMES:
            sums[k] += float(terms[k].item()) * bs

        pred = cls_logits.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += bs

        # MPS keeps freed tensors in a private pool. With dense PyG graphs
        # (~1500 nodes, ~58k edges per patient x batch_size) the pool grows
        # faster than it drains, which produced the "15.74 GiB allocated,
        # max 20.13 GiB" OOM on backward. Empty the cache between batches so
        # the next forward pass can allocate cleanly. No-op on CUDA/CPU.
        if device.type == "mps":
            torch.mps.empty_cache()

    n = total if total > 0 else 1
    out = {
        "loss": total_loss / n,
        # Legacy key names kept for training_results.json schema compatibility.
        "main_loss": sums["ce"] / n,
        "ce_loss": sums["ce"] / n,
        "deephit_loss": sums["deephit"] / n,
        "cox_loss": sums["cox"] / n,
        "ordinal_loss": sums["ordinal"] / n,
        "aux_loss": sums["aux_stage"] / n,
        "aux_stage_loss": sums["aux_stage"] / n,
        "accuracy": correct / n,
    }
    return out


def compute_gnn_grad_per_loss(
    model: HybridModel,
    batch,
    device: torch.device,
    active_losses,
    class_weights: torch.Tensor,
    label_smoothing: float = 0.1,
) -> dict:
    """For one batch, compute ||grad_L_k (gnn.params)|| for each active L_k.

    Runs ONE forward and N backwards with retain_graph=True (one per active
    loss), zeroing gradients between each. Model gradients are zeroed on
    exit -- caller can safely call optimizer.step() next on a fresh batch.

    Returns {loss_name: float grad_norm summed over gnn.* parameters}.
    """
    was_training = model.training
    model.train()
    model.zero_grad()

    main_criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )
    aux_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    ordinal_criterion = nn.SmoothL1Loss()

    gene_idx = getattr(batch, "gene_idx", None)
    cls_logits, aux_out, _, cox_out, ordinal_out, bin_logits = model(
        batch.x, batch.edge_index, batch.batch, batch.clinical,
        gene_idx=gene_idx,
    )
    terms = _compute_loss_terms(
        batch, cls_logits, aux_out, cox_out, ordinal_out, bin_logits,
        main_criterion, aux_criterion, ordinal_criterion,
        set(active_losses), device,
    )

    # Filter to only terms that actually have grad (active + non-zero).
    grad_terms = [(k, terms[k]) for k in active_losses if terms[k].requires_grad]

    grad_norms = {k: 0.0 for k in active_losses}
    for i, (name, val) in enumerate(grad_terms):
        model.zero_grad()
        retain = i < len(grad_terms) - 1
        val.backward(retain_graph=retain)
        g = 0.0
        for p in model.gnn.parameters():
            if p.grad is not None:
                g += float(p.grad.norm().item())
        grad_norms[name] = g

    model.zero_grad()
    if not was_training:
        model.eval()
    return grad_norms


@torch.no_grad()
def evaluate(model: HybridModel, loader, device) -> dict:
    """Evaluate model on validation data.

    Accuracy and AUC-ROC come from the CE-trained classifier head (cls_logits).
    The "classifier expected-bin" c-index comes from the DeepHit-trained
    survival-bin head (bin_logits) -- after the head decoupling, bin_logits
    are the output whose softmax is calibrated as a discrete-time event
    distribution, which is what the expected-bin score requires.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_cls_probs = []
    all_bin_probs = []
    all_cox = []
    all_ordinal = []
    all_os_time = []
    all_os_event = []
    total = 0

    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        batch = batch.to(device)

        gene_idx = getattr(batch, "gene_idx", None)
        cls_logits, _, _, cox_out, ordinal_out, bin_logits = model(
            batch.x, batch.edge_index, batch.batch, batch.clinical,
            gene_idx=gene_idx,
        )
        loss = criterion(cls_logits, batch.y)

        total_loss += loss.item() * batch.num_graphs
        total += batch.num_graphs

        cls_probs = F.softmax(cls_logits, dim=1)
        bin_probs = F.softmax(bin_logits, dim=1)
        pred = cls_logits.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        all_cls_probs.extend(cls_probs.cpu().numpy())
        all_bin_probs.extend(bin_probs.cpu().numpy())
        all_cox.extend(cox_out.cpu().numpy())
        all_ordinal.extend(ordinal_out.cpu().numpy())

        if hasattr(batch, "os_time") and batch.os_time is not None:
            all_os_time.extend(batch.os_time.cpu().numpy())
            all_os_event.extend(batch.os_event.cpu().numpy())

        if device.type == "mps":
            torch.mps.empty_cache()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_cls_probs = np.array(all_cls_probs)
    all_bin_probs = np.array(all_bin_probs)
    all_cox = np.array(all_cox)
    all_ordinal = np.array(all_ordinal)

    n = total if total > 0 else 1
    metrics = {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
    }

    # AUC-ROC (multi-class, one-vs-rest) from the CE classifier head.
    try:
        if len(np.unique(all_labels)) > 1:
            metrics["auc_roc"] = roc_auc_score(all_labels, all_cls_probs, multi_class="ovr", average="macro")
        else:
            metrics["auc_roc"] = 0.0
    except ValueError:
        metrics["auc_roc"] = 0.0

    # C-index (concordance index) — THE standard for survival (Zohari et al., 2025)
    # lifelines convention: higher predicted_scores = longer survival time.
    if all_os_time and all_os_event:
        os_time = np.array(all_os_time)
        os_event = np.array(all_os_event)
        valid = ~np.isnan(os_time) & ~np.isnan(os_event) & (os_time > 0)

        if valid.sum() > 10:
            # 1) Expected bin from the DeepHit-trained survival-bin head.
            #    Higher expected bin = longer survival (bins are ordered).
            #    Name kept as `c_index_cls` for historical compatibility with
            #    the training_results.json schema; the underlying source is
            #    now bin_logits, not cls_logits.
            predicted_survival = np.sum(
                all_bin_probs * np.arange(all_bin_probs.shape[1]), axis=1
            )
            # 2) Cox head log-hazard (higher hazard = shorter survival, so negate)
            cox_score = -all_cox
            # 3) Ordinal head score (trained on float label -> already higher = longer)
            ord_score = all_ordinal

            def _safe_cidx(score):
                try:
                    return concordance_index(os_time[valid], score[valid], os_event[valid])
                except Exception:
                    return 0.5

            c_cls = _safe_cidx(predicted_survival)
            c_cox = _safe_cidx(cox_score)
            c_ord = _safe_cidx(ord_score)

            metrics["c_index_cls"] = c_cls
            metrics["c_index_cox"] = c_cox
            metrics["c_index_ord"] = c_ord
            # Headline metric: the ordinal head specifically. Previously this
            # was `max(c_cls, c_cox, c_ord)` per fold, which cherry-picked
            # per-fold and produced an optimistic aggregate that couldn't be
            # compared to baselines. The ordinal head is chosen because:
            #   - it's trained directly on the discretized bin ordering,
            #   - its output is a scalar score that c-index consumes cleanly,
            #   - unlike Cox it doesn't need a ranking pass, so it's stable
            #     fold-over-fold when the event count is low.
            metrics["c_index"] = c_ord
        else:
            metrics["c_index"] = 0.5
            metrics["c_index_cls"] = 0.5
            metrics["c_index_cox"] = 0.5
            metrics["c_index_ord"] = 0.5

    return metrics


@torch.no_grad()
def extract_embeddings(model: HybridModel, loader, device) -> tuple:
    """Extract GNN embeddings for all patients in the loader.

    Used for Hybrid GAT -> RF pipeline (Palmal et al., 2024).

    Returns:
        (embeddings, labels, clinical_features)
    """
    model.eval()
    all_emb = []
    all_labels = []
    all_clinical = []

    for batch in loader:
        batch = batch.to(device)
        gene_idx = getattr(batch, "gene_idx", None)
        emb = model.extract_embeddings(
            batch.x, batch.edge_index, batch.batch, batch.clinical, gene_idx=gene_idx,
        )
        all_emb.append(emb.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        all_clinical.append(batch.clinical.cpu().numpy())

    return (
        np.concatenate(all_emb),
        np.concatenate(all_labels),
        np.concatenate(all_clinical),
    )


def train_calibrated_rf(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    config: dict,
) -> tuple:
    """Train Calibrated Random Forest on GNN embeddings.

    Reference: Palmal et al., 2024 — hybrid GCN+RF approach.

    Returns:
        (calibrated_rf_model, val_metrics)
    """
    rf_cfg = config["hybrid"]

    rf = RandomForestClassifier(
        n_estimators=rf_cfg["rf_n_estimators"],
        max_depth=rf_cfg.get("rf_max_depth"),
        min_samples_split=rf_cfg.get("rf_min_samples_split", 5),
        random_state=config["training"]["seed"],
        n_jobs=-1,
    )

    if rf_cfg.get("rf_calibrated", True):
        model = CalibratedClassifierCV(rf, cv=3, method="isotonic")
    else:
        model = rf

    model.fit(train_emb, train_labels)

    # Evaluate
    val_pred = model.predict(val_emb)
    val_probs = model.predict_proba(val_emb)

    metrics = {"accuracy": accuracy_score(val_labels, val_pred)}

    try:
        if len(np.unique(val_labels)) > 1:
            metrics["auc_roc"] = roc_auc_score(val_labels, val_probs, multi_class="ovr", average="macro")
    except ValueError:
        pass

    logger.info(f"RF metrics: {metrics}")
    return model, metrics


def train_fold(
    fold: int,
    model: HybridModel,
    dataset: BreastCancerGraphDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict:
    """Train one CV fold end-to-end (GAT + RF).

    Returns:
        Dictionary with best model, metrics, and RF model.
    """
    train_cfg = config["training"]
    abl_cfg = config.get("ablation", {}) or {}
    ablation_enabled = bool(abl_cfg.get("enabled", False))
    run_name = abl_cfg.get("run_name", "full")
    active_losses = list(abl_cfg.get("active_losses", list(LOSS_NAMES)))
    max_epochs = (
        int(abl_cfg.get("max_epochs", train_cfg["epochs"]))
        if ablation_enabled else int(train_cfg["epochs"])
    )
    dead_early_kill = bool(abl_cfg.get("dead_early_kill", False)) and ablation_enabled

    loss_weights = {
        "ce": 1.0,
        "deephit": float(train_cfg.get("deephit_aux_weight", 0.3)),
        "cox": float(train_cfg.get("cox_loss_weight", 0.5)),
        "ordinal": float(train_cfg.get("ordinal_loss_weight", 0.2)),
        "aux_stage": float(train_cfg.get("aux_loss_weight", 0.3)),
    }

    # Create data loaders (SMOTE on training set only -- skipped if dims too large)
    train_loader, val_loader, smote_applied = get_dataloaders(
        dataset, train_idx, val_idx,
        batch_size=train_cfg["batch_size"],
        smote=True,
        smote_strategy=config["data"]["smote_strategy"],
        seed=train_cfg["seed"],
    )

    # Class weighting: pick ONE of SMOTE or inverse-frequency weights, never both.
    # Phase 1 diagnosis -- val accuracy was stuck at 0.2419 (chance for 4 classes)
    # because SMOTE balanced the train distribution to 25/25/25/25 AND inverse-
    # frequency weights were still multiplying the loss. The combined effect
    # was to boost the minority class twice, driving the classifier to collapse
    # on whichever class dominated late in each epoch.
    train_labels = dataset.survival_labels[train_idx]
    if smote_applied:
        logger.info(
            "  SMOTE active -- using uniform class weights "
            "(avoid double-rebalancing)"
        )
        class_weights = torch.ones(
            int(np.nanmax(train_labels)) + 1, dtype=torch.float, device=device
        )
    else:
        cw = compute_class_weights(train_labels)
        class_weights = torch.tensor(cw, dtype=torch.float, device=device)
        logger.info(f"  Class weights (inverse-freq): {dict(enumerate(cw.tolist()))}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    # Early-stop metric: 'auc_roc' (maximize) is more stable than val_loss,
    # especially with the multi-task loss whose magnitude varies epoch to epoch.
    es_metric = train_cfg.get("early_stop_metric", "auc_roc")
    es_mode = "max" if es_metric in ("auc_roc", "accuracy", "c_index") else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=es_mode, factor=0.5, patience=10
    )

    best_score = -float("inf") if es_mode == "max" else float("inf")
    best_model_state = None
    patience_counter = 0
    history = defaultdict(list)

    # Val-set reference numbers for the [ABLATION] log line. The classifier
    # at uniform softmax has CE = ln(n_classes); the "majority" column on
    # val_acc is the accuracy a model would get by always predicting the
    # most common val-set class (e.g. SMOTE-trained models often beat this
    # on train but undershoot on val because val is still imbalanced).
    val_labels_arr = dataset.survival_labels[val_idx]
    val_labels_clean = val_labels_arr[~np.isnan(val_labels_arr)] if np.any(np.isnan(val_labels_arr)) else val_labels_arr
    if len(val_labels_clean) > 0:
        vals, counts = np.unique(val_labels_clean.astype(int), return_counts=True)
        majority_freq = float(counts.max()) / float(counts.sum())
    else:
        majority_freq = 0.0
    n_classes = len(config["data"].get("survival_labels", [0, 1, 2, 3]))
    uniform_ce = float(np.log(n_classes))

    logger.info(
        f"--- Fold {fold + 1}/{config['training']['cv_folds']} "
        f"(early-stop on val_{es_metric} {es_mode}, "
        f"ablation_enabled={ablation_enabled} run={run_name} "
        f"active_losses={active_losses} max_epochs={max_epochs}) ---"
    )

    # First train batch re-used each epoch for the per-loss GNN grad
    # decomposition. Taking the same batch every epoch removes one source
    # of noise from the grad-magnitude comparison across epochs.
    try:
        grad_probe_batch = next(iter(train_loader)).to(device)
    except StopIteration:
        grad_probe_batch = None

    dead_early_triggered = False

    for epoch in range(max_epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            active_losses=active_losses,
            loss_weights=loss_weights,
            label_smoothing=train_cfg.get("label_smoothing", 0.1),
            class_weights=class_weights,
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Learning rate scheduling follows the early-stop metric
        scheduler.step(val_metrics.get(es_metric, val_metrics["loss"]))

        # Per-loss GNN grad decomposition (single forward + N backwards on
        # a held-out probe batch). Cheap enough at once-per-epoch.
        grad_norms = {}
        if grad_probe_batch is not None:
            try:
                grad_norms = compute_gnn_grad_per_loss(
                    model, grad_probe_batch, device,
                    active_losses=active_losses,
                    class_weights=class_weights,
                    label_smoothing=train_cfg.get("label_smoothing", 0.1),
                )
            except Exception as e:
                logger.warning(f"  grad-norm probe failed: {e}")
                grad_norms = {}

        # Track history
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)
        for k, v in grad_norms.items():
            history[f"grad_{k}"].append(v)

        # Early stopping on chosen metric
        score = val_metrics.get(es_metric, val_metrics["loss"])
        improved = (score > best_score) if es_mode == "max" else (score < best_score)
        if improved:
            best_score = score
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # [ABLATION] line -- always emitted so the sweep log has one line
        # per epoch. `uniform` and `majority` are printed in-line so the
        # reader can judge "is the classifier still at chance?" at a glance.
        c_idx_ord = val_metrics.get("c_index_ord", val_metrics.get("c_index", 0.0))
        logger.info(
            f"[ABLATION {run_name}] fold={fold} epoch={epoch + 1:3d} "
            f"train_ce={train_metrics['ce_loss']:.4f} (uniform={uniform_ce:.4f}) "
            f"train_dh={train_metrics['deephit_loss']:.4f} "
            f"train_cox={train_metrics['cox_loss']:.4f} "
            f"train_ord={train_metrics['ordinal_loss']:.4f} "
            f"train_aux={train_metrics['aux_stage_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} (majority={majority_freq:.4f}) "
            f"val_auc={val_metrics.get('auc_roc', 0.0):.4f} "
            f"c_idx_ord={c_idx_ord:.4f}"
        )
        if grad_norms:
            grad_str = " ".join(f"{k}={v:.3e}" for k, v in grad_norms.items())
            logger.info(f"[GRAD {run_name}] fold={fold} epoch={epoch + 1:3d} {grad_str}")

        # Dead-early kill (ablation runs only): after epoch 5, if CE hasn't
        # moved off the uniform-softmax plateau and val_acc is still below
        # chance, there's no point burning the remaining compute.
        if (
            dead_early_kill
            and (epoch + 1) == 5
            and train_metrics["ce_loss"] > 1.35
            and val_metrics["accuracy"] < 0.25
        ):
            logger.warning(
                f"  [DEAD EARLY] run={run_name} fold={fold}: "
                f"train_ce={train_metrics['ce_loss']:.3f}>1.35 AND "
                f"val_acc={val_metrics['accuracy']:.3f}<0.25 at epoch 5 -- killing."
            )
            dead_early_triggered = True
            break

        if patience_counter >= train_cfg["patience"]:
            logger.info(f"  Early stopping at epoch {epoch + 1} (best {es_metric}={best_score:.4f})")
            break

    # Load best model (guard: if we never improved, best_model_state is None).
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final validation metrics
    final_metrics = evaluate(model, val_loader, device)
    logger.info(f"  Best val metrics: {final_metrics}")

    # Hybrid: Extract embeddings and train Calibrated RF. Skipped in ablation
    # mode -- the point of the sweep is GNN-loss isolation, not RF calibration.
    skip_rf = bool(abl_cfg.get("skip_rf", False)) and ablation_enabled
    rf_model = None
    rf_metrics = {}
    val_emb = None
    val_labels_out = None
    if skip_rf:
        logger.info("  [ABLATION] skip_rf=true -- not training Calibrated RF")
    else:
        logger.info("  Training Calibrated Random Forest on GNN embeddings...")
        train_emb, train_labels_emb, _ = extract_embeddings(model, train_loader, device)
        val_emb, val_labels_out, _ = extract_embeddings(model, val_loader, device)
        rf_model, rf_metrics = train_calibrated_rf(
            train_emb, train_labels_emb, val_emb, val_labels_out, config
        )

    return {
        "model_state": best_model_state,
        "gat_metrics": final_metrics,
        "rf_model": rf_model,
        "rf_metrics": rf_metrics,
        "history": dict(history),
        "val_emb": val_emb,
        "val_labels": val_labels_out,
        "dead_early": dead_early_triggered,
        "run_name": run_name,
    }


def run_training(config: dict, dataset_cache: dict = None) -> dict:
    """Run the complete training pipeline with cross-validation.

    `dataset_cache` (optional): pre-built dataset dict (from build_dataset()).
    If provided, the expensive KG build + per-patient embedding step is
    skipped. The ablation-sweep driver uses this to reuse the same dataset
    across all 4 runs.

    Respects `config.ablation` when `ablation.enabled=true`:
      - caps epochs at `max_epochs`
      - restricts the loss to `active_losses`
      - writes a run-specific log file to
        `results/ablation_{run_name}.log`
      - writes results JSON to `results/ablation_{run_name}_results.json`
      - skips RF training when `skip_rf=true`
    """
    seed = config["training"]["seed"]
    set_seed(seed)
    device_cfg = str(config.get("training", {}).get("device", "")).lower()
    if device_cfg in ("cpu", "cuda", "mps"):
        device = torch.device(device_cfg)
    else:
        device = get_device()
    logger.info(f"Using device: {device}")

    abl_cfg = config.get("ablation", {}) or {}
    ablation_enabled = bool(abl_cfg.get("enabled", False))
    run_name = abl_cfg.get("run_name", "full")

    results_dir = config["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    # Per-run log file in ablation mode. A file handler is attached to the
    # root logger so everything (dataset build + per-epoch + grad lines)
    # lands in one place, and detached afterwards so subsequent runs don't
    # accumulate handlers.
    log_handler = None
    if ablation_enabled:
        log_path = os.path.join(results_dir, f"ablation_{run_name}.log")
        log_handler = logging.FileHandler(log_path, mode="w")
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logging.getLogger().addHandler(log_handler)
        logger.info(f"[ABLATION] writing per-run log to {log_path}")

    # Build (or reuse) dataset
    if dataset_cache is not None:
        data = dataset_cache
        logger.info("Using cached dataset (skip KG build)")
    else:
        logger.info("=" * 60)
        logger.info("Building dataset...")
        logger.info("=" * 60)
        data = build_dataset(config)
    dataset = data["dataset"]
    splits = data["splits"]
    clinical_dim = data["clinical_dim"]
    num_genes = data["n_genes"]

    all_fold_results = []
    all_gat_metrics = defaultdict(list)
    all_rf_metrics = defaultdict(list)

    # Cap fold iteration to training.cv_folds. The dataset cache may have
    # been pre-built with more folds than this run needs (e.g. when the
    # ablation-sweep driver reuses a cv_folds=2 dataset_cache for a
    # cv_folds=1 benchmark).
    cv_folds_cap = int(config.get("training", {}).get("cv_folds", len(splits)))
    splits_to_run = list(splits)[:cv_folds_cap]
    for fold, (train_idx, val_idx) in enumerate(splits_to_run):
        # Build fresh model for each fold
        model = build_model(
            config, clinical_dim=clinical_dim, device=str(device),
            num_genes=num_genes,
        )

        fold_result = train_fold(fold, model, dataset, train_idx, val_idx, config, device)
        all_fold_results.append(fold_result)

        for k, v in fold_result["gat_metrics"].items():
            all_gat_metrics[k].append(v)
        if fold_result.get("rf_metrics"):
            for k, v in fold_result["rf_metrics"].items():
                all_rf_metrics[k].append(v)

        # Save fold model only when we actually have one.
        if fold_result.get("model_state") is not None:
            if ablation_enabled:
                model_path = os.path.join(
                    results_dir, f"ablation_{run_name}_fold{fold}.pt"
                )
            else:
                model_path = os.path.join(results_dir, f"model_fold{fold}.pt")
            torch.save(fold_result["model_state"], model_path)

    # Aggregate metrics
    logger.info("=" * 60)
    logger.info("Cross-Validation Results (GAT)")
    logger.info("=" * 60)
    gat_summary = {}
    for k, values in all_gat_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        gat_summary[k] = {"mean": mean, "std": std}
        logger.info(f"  {k}: {mean:.4f} +/- {std:.4f}")

    rf_summary = {}
    if all_rf_metrics:
        logger.info("=" * 60)
        logger.info("Cross-Validation Results (Calibrated RF)")
        logger.info("=" * 60)
        for k, values in all_rf_metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            rf_summary[k] = {"mean": mean, "std": std}
            logger.info(f"  {k}: {mean:.4f} +/- {std:.4f}")

    # Save results
    results = {
        "gat_summary": gat_summary,
        "rf_summary": rf_summary,
        "run_name": run_name,
        "ablation_enabled": ablation_enabled,
        "active_losses": list(abl_cfg.get("active_losses", [])),
        "fold_results": [
            {
                "gat_metrics": fr["gat_metrics"],
                "rf_metrics": fr.get("rf_metrics", {}),
                "history": fr["history"],
                "dead_early": fr.get("dead_early", False),
                "run_name": fr.get("run_name", run_name),
            }
            for fr in all_fold_results
        ],
        "config": config,
    }

    if ablation_enabled:
        out_path = os.path.join(
            results_dir, f"ablation_{run_name}_results.json"
        )
    else:
        out_path = os.path.join(results_dir, "training_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Detach per-run file handler so repeated run_training calls don't
    # accumulate writes to stale logs.
    if log_handler is not None:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()

    return {
        "fold_results": all_fold_results,
        "gat_summary": gat_summary,
        "rf_summary": rf_summary,
        "dataset": data,
        "run_name": run_name,
    }


if __name__ == "__main__":
    config = load_config()
    results = run_training(config)
