"""Storage accounting oracles for task memory, independent of region planning."""

import torch


def test_meta_memory_tracks_aliases_saved_gradients_and_real_cuda_peaks():
    from pllm.execution.memory import measure

    source = torch.empty(1024, device="meta")
    assert measure(lambda: source.view(32, 32)).workspace == 0

    def views():
        value = source.clone()
        return value, value.view(32, 32), value[1:]

    assert measure(views).workspace == 4096

    def temporary():
        source.clone()
        return source.clone()

    assert measure(temporary).workspace == 4096

    def program(x):
        leaf = x.detach().requires_grad_(True)
        loss = leaf.sin().square().mean()
        return loss.detach(), torch.autograd.grad(loss, leaf)[0]

    bound = measure(lambda: program(source)).workspace
    assert bound > source.numel() * source.element_size()
    if torch.cuda.is_available():
        actual = torch.randn(source.shape, device="cuda")
        program(actual)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = program(actual)
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() - baseline <= bound
        assert result[1].shape == actual.shape
