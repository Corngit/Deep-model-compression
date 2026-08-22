import argparse
import time

import numpy as np
import torch

from datasets.cifar100 import get_cifar100_loaders
from models.model_factory import create_model

def count_parameters(model):
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params, trainable_params

def get_model_size_mb(model):

    size_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    size_bytes += sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )

    size_mb = size_bytes / (1024 ** 2)

    return size_mb

def load_model_from_checkpoint(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    saved_args = checkpoint["args"]

    model_name = saved_args["model"]

    model = create_model(
        model_name=model_name,
        num_classes=100,
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, model_name

@torch.inference_mode()
def evaluate_accuracy(
    model,
    dataloader,
    device,
):
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return correct / total

@torch.inference_mode()
def benchmark_latency(
    model,
    device,
    batch_size=1,
    warmup_iters=10,
    benchmark_iters=50,
):
    dummy_input = torch.randn(
        batch_size,
        3,
        224,
        224,
        device=device,
    )

    # Warm-up
    for _ in range(warmup_iters):
        _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []

    for _ in range(benchmark_iters):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        latencies.append(latency_ms)

    average_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)

    throughput = (
        batch_size
        / (average_latency / 1000)
    )

    return {
        "average_latency_ms":
            average_latency,

        "p50_latency_ms":
            p50_latency,

        "p95_latency_ms":
            p95_latency,

        "throughput_images_per_sec":
            throughput,
    }

def parse_args():

    parser = argparse.ArgumentParser(
        description="Benchmark trained models."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=50,
    )

    return parser.parse_args()

def main():

    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, model_name = (
        load_model_from_checkpoint(
            args.checkpoint,
            device,
        )
    )

    total_params, trainable_params = (
        count_parameters(model)
    )

    model_size_mb = get_model_size_mb(model)

    _, _, test_loader = (
        get_cifar100_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(
                device.type == "cuda"
            ),
            max_test_samples=(
                args.max_test_samples
            ),
        )
    )

    accuracy = evaluate_accuracy(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    latency_results = benchmark_latency(
        model=model,
        device=device,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        benchmark_iters=(
            args.benchmark_iters
        ),
    )

    print()
    print("=" * 60)
    print("Model Benchmark")
    print("=" * 60)

    print(
        f"Model            : "
        f"{model_name}"
    )

    print(
        f"Device           : "
        f"{device}"
    )

    print(
        f"Parameters       : "
        f"{total_params / 1e6:.2f} M"
    )

    print(
        f"Trainable Params : "
        f"{trainable_params / 1e6:.2f} M"
    )

    print(
        f"Weight Size      : "
        f"{model_size_mb:.2f} MB"
    )

    print(
        f"Test Accuracy    : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average Latency  : "
        f"{latency_results['average_latency_ms']:.2f} ms"
    )

    print(
        f"P50 Latency      : "
        f"{latency_results['p50_latency_ms']:.2f} ms"
    )

    print(
        f"P95 Latency      : "
        f"{latency_results['p95_latency_ms']:.2f} ms"
    )

    print(
        f"Throughput       : "
        f"{latency_results['throughput_images_per_sec']:.2f} images/s"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()