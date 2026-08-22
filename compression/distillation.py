import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=4.0,
    alpha=0.5,
):
    hard_loss = F.cross_entropy(
        student_logits,
        labels,
    )

    student_log_probs = F.log_softmax(
        student_logits / temperature,
        dim=1,
    )

    teacher_probs = F.softmax(
        teacher_logits / temperature,
        dim=1,
    )

    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    )

    soft_loss = (
        soft_loss
        * temperature
        * temperature
    )

    total_loss = (
        alpha * hard_loss
        +
        (1 - alpha) * soft_loss
    )

    return total_loss