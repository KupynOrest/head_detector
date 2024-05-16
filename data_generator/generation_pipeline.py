import torch
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, DPMSolverMultistepScheduler, AutoencoderKL


def get_pipeline() -> StableDiffusionXLAdapterPipeline:
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-openpose-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
    ).to("cuda")

    # load euler_a scheduler
    model_id = 'SG161222/RealVisXL_V4.0'
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, adapter=adapter, torch_dtype=torch.float16, variant="fp16",
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()
    pipe = pipe.to("cuda")
    return pipe
