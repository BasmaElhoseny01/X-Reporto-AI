import torch


def cuda_memory_info():
    num_cuda_devices = torch.cuda.device_count()
    for i in range(num_cuda_devices):
        device = torch.cuda.get_device_properties(i)
        total_memory = device.total_memory / 1024**3  # Total memory available on the device
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3  # Memory currently in use by tensors
        reserved_memory = torch.cuda.memory_reserved(i) / 1024**3  # Total memory reserved by PyTorch
        remaining_memory_current = total_memory - allocated_memory  # Remaining memory that is currently available for allocation
        remaining_memory_potential = total_memory - reserved_memory  # Remaining memory that can potentially be allocated
        print(f"Device {i}:")
        print(f"  Name: {device.name}")
        print(f"  Total Memory: {total_memory:.2f} GB")
        print(f"  Allocated Memory: {allocated_memory:.2f} GB")
        print(f"  Reserved Memory: {reserved_memory:.2f} GB")
        print(f"  Remaining Memory (Current): {remaining_memory_current:.2f} GB")
        print(f"  Remaining Memory (Potential): {remaining_memory_potential:.2f} GB")

import time

if __name__ == "__main__":
    
    cuda_memory_info()

    # # Allocate some tensors on GPU
    tensor1 = torch.randn(100000, 1000).cuda()
    tensor2 = torch.randn(20000, 20000).cuda()


    cuda_memory_info()

    time.sleep(10)
    print("Deallocating\n\n")
    del tensor1,tensor2
    torch.cuda.empty_cache()
    

    cuda_memory_info()

    time.sleep(10)
    print("New Allocation\n\n")
    # Specify the size of the tensor in bytes (3GB)
    tensor_size_bytes = 3 * 1024**3  # 3GB in bytes

    # Determine the number of elements in the tensor
    num_elements = tensor_size_bytes // 4  # Assuming 32-bit floating-point numbers (4 bytes per element)

    # Generate random data
    random_data = torch.randn((num_elements,)).cuda()

    cuda_memory_info()
    print("Done")


    while True:
        pass


# if __name__ == "__main__":
    
#     cuda_memory_info()

#     # # Allocate some tensors on GPU
#     tensor1 = torch.randn(100000, 1000).cuda()
#     tensor2 = torch.randn(20000, 20000).cuda()


#     cuda_memory_info()

#     time.sleep(10)
#     print("Deallocating\n\n")
#     tensor1=tensor1.cpu()
#     tensor2=tensor2.cpu()
#     # del tensor1,tensor2

#     cuda_memory_info()


#     while True:
#         pass



