"""Memory locality for the Qwen megakernel: which memory side each SM is on, and weights placed
on known sides.

On GPUs with two memory locality domains, an SM reads its own domain's HBM faster, so a
bandwidth-bound row streams faster on an SM of its weights' domain. Two driver features (CUDA
13.4 C API; the Python bindings do not expose them yet) are reached through a small inline
extension, built on first use:

- ``sm_sides(device)``: each SM's side. Green contexts split the SMs by locality domain; the SMs
  in neither domain are classified by the latency of a cold load from a side-0 page.
- ``striped_empty(nbytes, device)``: a uint8 tensor whose 2 MiB page P (from a 4 MiB-aligned base)
  lives on side P % 2, so byte offset o of the tensor is on side ``(o >> 21) & 1``.
"""

from functools import cache

import torch

PAGE_BYTES = 2 << 20

_CPP = r"""
#include <torch/extension.h>
std::vector<int64_t> sm_sides(int64_t device);
torch::Tensor striped_empty(int64_t nbytes, int64_t device);
"""

_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <functional>
#include <vector>

#define CU(x) do { CUresult e_ = (x); if (e_) { const char* s_; cuGetErrorString(e_, &s_); TORCH_CHECK(false, #x, ": ", s_); } } while (0)
#define RT(x) do { cudaError_t e_ = (x); TORCH_CHECK(e_ == cudaSuccess, #x, ": ", cudaGetErrorString(e_)); } while (0)

constexpr size_t kPage = size_t(2) << 20;

// Runs its cleanup when the scope exits, by return or by a TORCH_CHECK throw.
struct Defer {
  std::function<void()> fn;
  ~Defer() { fn(); }
};
// Median cold-load latency to side 0 above this: side 1. Rubin (sm_107, 2026-11): the SMs green
// contexts leave unassigned measured ~670 cycles (near) or ~1075 (far).
constexpr unsigned kFarCycles = 890;

__device__ unsigned smid() { unsigned id; asm volatile("mov.u32 %0, %%smid;" : "=r"(id)); return id; }
__global__ void record_smid(int* side, int domain) { if (threadIdx.x == 0) side[smid()] = domain; }
__global__ void time_unassigned(const char* side0_page, const int* side, unsigned* latency) {
  unsigned sm = smid();
  if (threadIdx.x || side[sm] >= 0 || atomicCAS(&latency[sm], 0u, 1u)) return;
  unsigned samples[9];
  for (int i = 0; i < 9; ++i) {
    // 1 KiB apart: distinct lines for every (SM, sample) of up to 227 SMs in one page.
    const unsigned* addr = (const unsigned*)(side0_page + (size_t(sm) * 9 + i) * 1024 % kPage);
    unsigned value;
    unsigned long long start, stop = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(addr) : "memory");
    asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 1; @p mov.u64 %0, %%clock64; }" : "+l"(stop) : "r"(value) : "memory");
    samples[i] = unsigned(stop - start);
  }
  for (int i = 1; i < 9; ++i)
    for (int j = i; j > 0 && samples[j] < samples[j - 1]; --j) { unsigned t = samples[j]; samples[j] = samples[j - 1]; samples[j - 1] = t; }
  latency[sm] = samples[4];
}

// Maps [base, base + bytes) page by page; on failure unmaps what it mapped and rethrows.
static void map_pages(CUdeviceptr base, size_t bytes, int device) {
  size_t mapped = 0;
  bool done = false;
  Defer undo{[&] { if (!done && mapped) cuMemUnmap(base, mapped); }};
  for (size_t page = 0; page < bytes / kPage; ++page) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
    prop.location.localized.deviceId = device;
    prop.location.localized.localityDomainId = unsigned(((base + page * kPage) >> 21) & 1);
    CUmemGenericAllocationHandle handle;
    CU(cuMemCreate(&handle, kPage, &prop, 0));
    CUresult mapped_page = cuMemMap(base + page * kPage, kPage, 0, handle, 0);
    cuMemRelease(handle);  // the mapping, if any, keeps the memory
    CU(mapped_page);
    mapped += kPage;
  }
  CUmemAccessDesc access{};
  access.location = {CU_MEM_LOCATION_TYPE_DEVICE, device};
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU(cuMemSetAccess(base, bytes, &access, 1));
  done = true;
}

torch::Tensor striped_empty(int64_t nbytes, int64_t device) {
  c10::cuda::CUDAGuard guard(device);
  RT(cudaFree(0));
  size_t bytes = (size_t(nbytes) + 2 * kPage - 1) / (2 * kPage) * (2 * kPage);
  CUdeviceptr base;
  CU(cuMemAddressReserve(&base, bytes, 2 * kPage, 0, 0));
  bool owned = true;  // until the tensor's deleter takes the mapping
  Defer release{[&] { if (owned) cuMemAddressFree(base, bytes); }};
  map_pages(base, bytes, int(device));
  auto deleter = [base, bytes, device](void*) {
    c10::cuda::CUDAGuard guard(device);
    cuMemUnmap(base, bytes);
    cuMemAddressFree(base, bytes);
  };
  auto out = torch::from_blob((void*)base, {nbytes}, deleter,
                              torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device));
  owned = false;
  return out;
}

std::vector<int64_t> sm_sides(int64_t device) {
  c10::cuda::CUDAGuard guard(device);
  RT(cudaFree(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, int(device)));
  int domains = 0, sms = 0;
  CU(cuDeviceGetAttribute(&domains, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, dev));
  RT(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, int(device)));
  TORCH_CHECK(domains == 2, "expected 2 locality domains, got ", domains);
  int* side = nullptr;
  unsigned* latency = nullptr;
  void* scratch = nullptr;
  CUdeviceptr page = 0;
  bool page_mapped = false;
  Defer free_all{[&] {
    cudaFree(scratch);
    cudaFree(side);
    cudaFree(latency);
    if (page_mapped) cuMemUnmap(page, 2 * kPage);
    if (page) cuMemAddressFree(page, 2 * kPage);
  }};
  RT(cudaMalloc(&side, sms * sizeof(int)));
  RT(cudaMemset(side, 0xff, sms * sizeof(int)));
  RT(cudaMalloc(&latency, sms * sizeof(unsigned)));
  RT(cudaMemset(latency, 0, sms * sizeof(unsigned)));
  RT(cudaDeviceSynchronize());
  CUdevResource all, split[2], rest;
  CU(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
  CU_DEV_SM_RESOURCE_GROUP_PARAMS params[2] = {};
  for (int d = 0; d < 2; ++d) {
    params[d].flags = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
    params[d].localityDomainId = d;
  }
  CU(cuDevSmResourceSplit(split, 2, &all, &rest, 0, params));
  for (int d = 0; d < 2; ++d) {
    CUdevResourceDesc desc;
    CUgreenCtx green = nullptr;
    CUcontext ctx;
    CUstream stream = nullptr;
    bool pushed = false;
    Defer leave{[&] {
      if (stream) cuStreamDestroy(stream);
      if (pushed) cuCtxPopCurrent(&ctx);
      if (green) cuGreenCtxDestroy(green);
    }};
    CU(cuDevResourceGenerateDesc(&desc, &split[d], 1));
    CU(cuGreenCtxCreate(&green, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CU(cuCtxFromGreenCtx(&ctx, green));
    CU(cuCtxPushCurrent(ctx));
    pushed = true;
    CU(cuGreenCtxStreamCreate(&stream, green, CU_STREAM_NON_BLOCKING, 0));
    record_smid<<<sms * 16, 32, 0, stream>>>(side, d);
    RT(cudaGetLastError());
    CU(cuStreamSynchronize(stream));
  }
  // A side-0 page for the latency probe; scratch writes push it out of L2 first.
  CU(cuMemAddressReserve(&page, 2 * kPage, 2 * kPage, 0, 0));
  map_pages(page, 2 * kPage, int(device));
  page_mapped = true;
  int l2 = 0;
  RT(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, int(device)));
  RT(cudaMalloc(&scratch, size_t(l2) * 8));
  RT(cudaMemset((void*)page, 0x5a, kPage));
  RT(cudaMemset(scratch, 1, size_t(l2) * 8));
  time_unassigned<<<sms * 16, 32>>>((const char*)page, side, latency);
  RT(cudaDeviceSynchronize());
  std::vector<int> host(sms);
  std::vector<unsigned> cycles(sms);
  RT(cudaMemcpy(host.data(), side, sms * sizeof(int), cudaMemcpyDeviceToHost));
  RT(cudaMemcpy(cycles.data(), latency, sms * sizeof(unsigned), cudaMemcpyDeviceToHost));
  std::vector<int64_t> out(sms);
  for (int s = 0; s < sms; ++s) {
    TORCH_CHECK(host[s] >= 0 || cycles[s] > 1, "SM ", s, " was never probed");
    out[s] = host[s] >= 0 ? host[s] : int64_t(cycles[s] > kFarCycles);
  }
  return out;
}
"""


@cache
def _extension():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        "qwen_megakernel_locality",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["sm_sides", "striped_empty"],
        extra_ldflags=["-lcuda"],
        extra_cuda_cflags=["-O2"],
    )


@cache
def sm_sides(device: int) -> tuple[int, ...]:
    """Each SM's memory side (0 or 1) on CUDA ``device``."""
    return tuple(_extension().sm_sides(device))


def striped_copy(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` copied into striped memory (``striped_empty``), same shape and dtype."""
    raw = _extension().striped_empty(tensor.numel() * tensor.element_size(), tensor.device.index)
    out = raw.view(tensor.dtype).view(tensor.shape)
    out.copy_(tensor)
    return out


def byte_side(offset: int) -> int:
    """The side of byte ``offset`` of a striped tensor."""
    return (offset // PAGE_BYTES) & 1
