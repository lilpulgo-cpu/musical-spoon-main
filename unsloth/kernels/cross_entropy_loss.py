import torch
from transformers.models.llama.modeling_llama import logger

# Utilidades
MAX_FUSED_SIZE = 65536  # 2**16

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping=0):
        n_rows, vocab_size = logits.shape
        losses = torch.empty(n_rows, dtype=torch.float32)

        # Calcular Cross Entropy Loss de manera tradicional
        logsumexp = torch.logsumexp(logits, dim=-1)
        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.logit_softcapping = logit_softcapping

        loss = logsumexp - logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        loss.masked_fill_(labels == -100, 0)  # Ignorar los padding tokens
        ctx.save_for_backward(logits, logsumexp, labels)
        return loss

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        vocab_size = logits.size(-1)

        # Calcular softmax
        exp_logits = torch.exp(logits - logsumexp.unsqueeze(-1))
        softmax = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

        dlogits = dlosses.unsqueeze(-1) * (softmax - (labels.unsqueeze(-1) == torch.arange(vocab_size, device=labels.device).unsqueeze(0)))

        return dlogits, None, None


def fast_cross_entropy_loss(logits, labels, logit_softcapping=0):
    """
    Calcula la pérdida de entropía cruzada usando una implementación simplificada, sin dependencias de Triton ni CUDA.
    
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))

    loss = Fast_CrossEntropyLoss.apply(
        logits.view(batch * seq_len, d),
        labels.view(-1),
        logit_softcapping,
    )
    n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items
