from lib.models.SD import SDModel
import torch

# Try mirror first (avoids 404; stabilityai deprecated). Fall back to legacy for users with cached models.
SD21_BASE_MODEL_IDS = [
    "Manojb/stable-diffusion-2-1-base",  # mirror (stabilityai deprecated)
    "stabilityai/stable-diffusion-2-1-base",
]


class SD21BaseModel(SDModel):
    def __init__(self, device="cuda", dtype=torch.float16):
        last_error = None
        for model_id in SD21_BASE_MODEL_IDS:
            try:
                super().__init__(model_id=model_id, device=device, dtype=dtype)
                return
            except Exception as e:
                last_error = e
                continue
        raise last_error

    def _get_noise_pred(self, latent_model_input, timestep, encoder_hidden_states):
        """
        Get noise prediction from SD 2.1 UNet (converting v-prediction to epsilon).
        """
        # # Get v-prediction from model
        # v_prediction = self.unet(
        #     latent_model_input, timestep, encoder_hidden_states=encoder_hidden_states
        # ).sample

        # # Get alpha and beta values for current timestep
        # alpha_prod_t = self.reference_scheduler.alphas_cumprod[timestep - 1]
        # beta_prod_t = 1 - alpha_prod_t

        # # Convert v-prediction to epsilon (noise) prediction
        # noise_pred = (alpha_prod_t ** 0.5) * v_prediction + (
        #     beta_prod_t ** 0.5
        # ) * latent_model_input

        # return noise_pred

        return self.unet(
            latent_model_input, timestep, encoder_hidden_states=encoder_hidden_states
        ).sample
