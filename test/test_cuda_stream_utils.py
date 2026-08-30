import pytest
import torch

from attn_gym.utils import fork_join_streams

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_fork_join_streams_orders_and_returns_both_outputs() -> None:
    current_stream = torch.cuda.current_stream()
    side_stream = torch.cuda.Stream()

    def side_work() -> tuple[torch.Tensor, torch.Tensor, str]:
        output = torch.zeros(1, device="cuda")
        torch.cuda._sleep(1_000_000)
        return output.fill_(3), output.new_full((1,), 4), "side"

    current_output, side_output = fork_join_streams(
        current_stream,
        side_stream,
        side_work,
        lambda: (torch.full((1,), 2, device="cuda"), "current"),
    )

    torch.testing.assert_close(current_output[0], torch.tensor([2], device="cuda"))
    assert current_output[1] == "current"
    torch.testing.assert_close(side_output[0], torch.tensor([3.0], device="cuda"))
    torch.testing.assert_close(side_output[1], torch.tensor([4.0], device="cuda"))
    assert side_output[2] == "side"
    assert torch.cuda.current_stream() == current_stream


def test_fork_join_streams_joins_when_side_work_raises() -> None:
    current_stream = torch.cuda.current_stream()
    side_stream = torch.cuda.Stream()
    marker = torch.zeros(1, device="cuda")

    def fail() -> torch.Tensor:
        torch.cuda._sleep(1_000_000)
        marker.fill_(1)
        raise RuntimeError("side work failed")

    with pytest.raises(RuntimeError, match="side work failed"):
        fork_join_streams(
            current_stream,
            side_stream,
            fail,
            lambda: torch.zeros(1, device="cuda"),
        )

    torch.testing.assert_close(marker, torch.ones_like(marker))


def test_fork_join_streams_joins_when_current_work_raises() -> None:
    current_stream = torch.cuda.current_stream()
    side_stream = torch.cuda.Stream()
    marker = torch.zeros(1, device="cuda")

    def side_work() -> torch.Tensor:
        torch.cuda._sleep(1_000_000)
        return marker.fill_(1)

    def fail() -> torch.Tensor:
        raise RuntimeError("current work failed")

    with pytest.raises(RuntimeError, match="current work failed"):
        fork_join_streams(current_stream, side_stream, side_work, fail)

    torch.testing.assert_close(marker, torch.ones_like(marker))
