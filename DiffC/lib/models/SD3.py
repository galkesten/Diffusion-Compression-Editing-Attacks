import torch
from diffusers import StableDiffusion3Pipeline
from lib.models.latent_noise_prediction_model import LatentNoisePredictionModel
from lib.diffc.utils.alpha_beta import get_alpha_prod_and_beta_prod
import numpy as np

# import lovely_tensors as lt
# lt.monkey_patch()


def sigma_to_snr(sigma):
    return (1 - sigma) / sigma


def get_ot_flow_to_ddpm_factor(snr):
    OT_flow_noise_sigma = 1 / (snr + 1)

    alpha_cumprod = snr ** 2 / (snr ** 2 + 1)
    DDPM_noise_sigma = torch.sqrt(1 - alpha_cumprod)

    ot_flow_to_ddpm_factor = DDPM_noise_sigma / OT_flow_noise_sigma

    return ot_flow_to_ddpm_factor


class SD3Model(LatentNoisePredictionModel):
    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        device="cuda",
        dtype=torch.float16,
    ):
        """
        Initialize the SD3 model.
        
        Args:
            model_id (str): HuggingFace model ID
            device (str): Device to run the model on ("cuda" or "cpu")
            dtype (torch.dtype): Data type for model parameters
        """
        self.device = device
        self.dtype = dtype

        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
        self.vae = self.pipe.vae

        # Pre-compute SNR values for timesteps
        sigmas = np.arange(1000) / 1000 # Default 1000 timesteps
        self.snr_values = torch.tensor(
            [sigma_to_snr(sigma) for sigma in sigmas], device=device
        )

        # Initialize configuration attributes
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None
        self.latent_image_ids = None
        self.guidance_scale = None
        self.image_width = None
        self.image_height = None
    
    # def enable_model_cpu_offload(self, gpu_id=None, device="cuda"):
    #     """
    #     Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance.
    #
    #     Args:
    #         gpu_id (int, optional): The ID of the GPU to use. Defaults to None.
    #         device (str, optional): The device to use. Defaults to "cuda".
    #     """
    #     torch_device = torch.device(device)
    #     if gpu_id is not None and torch_device.index is not None:
    #         raise ValueError(
    #             f"You have passed both `gpu_id`={gpu_id} and an index in `device`={device}. "
    #             "Please specify only one."
    #         )
    #
    #     # Set the GPU ID to use
    #     self._offload_gpu_id = gpu_id or torch_device.index or 0
    #     device = torch.device(f"{torch_device.type}:{self._offload_gpu_id}")
    #
    #     # First move everything to CPU
    #     self.to("cpu")
    #     if hasattr(torch.cuda, "empty_cache"):
    #         torch.cuda.empty_cache()
    #
    #     # Set up hooks for the models in sequence
    #     self._all_hooks = []
    #     hook = None
    #
    #     # Define sequence of models to offload (matching the Flux pipeline's sequence)
    #     model_sequence = [
    #         ("text_encoder", self.text_encoder),
    #         ("text_encoder_2", self.text_encoder_2),
    #         ("transformer", self.transformer),
    #         ("vae", self.vae)
    #     ]
    #
    #     # Set up CPU offloading hooks for each model in sequence
    #     for name, model in model_sequence:
    #         if not isinstance(model, torch.nn.Module):
    #             continue
    #
    #         # Set up CPU offloading with hook
    #         _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
    #         self._all_hooks.append(hook)

    def to(self, device, silence_dtype_warnings=True):
        """
        Moves all models to the specified device.
        
        Args:
            device (str or torch.device): Device to move models to
            silence_dtype_warnings (bool, optional): Whether to silence dtype warnings. Defaults to True.
        """
        raise NotImplementedError

        # self.device = device
        #
        # # Move all models to device
        # if hasattr(self, "text_encoder"):
        #     self.text_encoder.to(device)
        # if hasattr(self, "text_encoder_2"):
        #     self.text_encoder_2.to(device)
        # if hasattr(self, "transformer"):
        #     self.transformer.to(device)
        # if hasattr(self, "vae"):
        #     self.vae.to(device)
        #
        # return self

    def get_timestep_snr(self, timestep):
        """Return the SNR value for a given timestep."""
        if timestep == 0:
            return torch.inf
        return self.snr_values[timestep - 1]

    def image_to_latent(self, img_pt):
        """
        Convert input image tensor to latent representation.
        """
        if img_pt.dim() == 3:
            img_pt = img_pt.unsqueeze(0)

        # Move input to correct device and type
        img_pt = img_pt.to(dtype=self.dtype)

        # Encode image to latent space
        vae_latent = self.vae.encode(img_pt * 2 - 1).latent_dist.sample()
        vae_latent = vae_latent * self.vae.config.scaling_factor

        return vae_latent


    def latent_to_image(self, latent):
        """Convert packed latent representation back to image."""
        # Scale and shift
        vae_latent = latent / self.vae.config.scaling_factor
        
        # Decode to image space
        with torch.no_grad():
            image = self.vae.decode(vae_latent).sample
            
        return (image / 2 + 0.5).clamp(0, 1).detach()

    def configure(self, prompt, prompt_guidance, image_width, image_height):
        """
        Configure model with prompt and parameters.
        
        Args:
            prompt (str or List[str]): Text prompt(s)
            prompt_guidance (float): Classifier-free guidance scale
            image_width (int): Output image width
            image_height (int): Output image height
        """
        self.pipe._guidance_scale = prompt_guidance

        (prompt_embeds, _, pooled_prompt_embeds, _,) = self.pipe.encode_prompt(prompt=prompt, prompt_2=None,
                                                                               prompt_3=None, negative_prompt="",
                                                                               do_classifier_free_guidance=self.pipe.do_classifier_free_guidance,
                                                                               negative_pooled_prompt_embeds=None,
                                                                               device=self.device)
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds


    def predict_noise(self, noisy_latent, timestep):
        """
        Predict noise in the latent at given timestep.
        
        Args:
            noisy_latent (torch.Tensor): Noisy latent tensor
            timestep (int): Current timestep
            
        Returns:
            torch.Tensor: Predicted noise in DDPM space
        """
        # Get current SNR for scaling
        snr = self.get_timestep_snr(timestep)

        # Get scaling factor to convert between DDPM and OT flow spaces
        ot_flow_to_ddpm_factor = get_ot_flow_to_ddpm_factor(snr)

        # Convert DDPM space latent to OT flow space
        ot_flow_latent = noisy_latent / ot_flow_to_ddpm_factor

        with torch.no_grad():
            ot_flow_noise_pred = self.pipe.transformer(
                hidden_states=ot_flow_latent,
                timestep=torch.tensor([timestep], device=self.device),
                pooled_projections=self.pooled_prompt_embeds,
                encoder_hidden_states=self.prompt_embeds,
                return_dict=False,
            )[0].to(torch.float32)

        # TODO: this code is needlessly complicated, because I wanted to avoid doing math.
        # clean it up.
        # TODO: calculate x0 hat in OT flow space:
        sigma = 1 / (snr + 1)
        alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(snr)
        x0_hat = ot_flow_latent - sigma * ot_flow_noise_pred

        # back-calculate the DDPM noise pred from noisy_latent and x0_hat
        ddpm_noise_pred = (noisy_latent - alpha_prod_t**0.5 * x0_hat) / beta_prod_t**0.5
        # Convert prediction back to DDPM space
        #ddpm_noise_pred = ot_flow_noise_pred * ot_flow_to_ddpm_factor * (alpha_prod_t ** 0.5)

        return ddpm_noise_pred.to(noisy_latent.dtype)
