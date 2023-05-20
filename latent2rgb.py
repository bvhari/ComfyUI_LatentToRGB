import torch


class LatentToRGB:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", )}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "latent_to_rgb"

    CATEGORY = "latent"

    def latent_to_rgb(self, samples):
        latent_rgb_factors = torch.tensor([
                #  R      G      B
                [0.298, 0.207, 0.208],     # L1
                [0.187, 0.286, 0.173],     # L2
                [-0.158, 0.189, 0.264],    # L3
                [-0.184, -0.271, -0.473],  # L4
                ], device="cpu")
    
        rgb = torch.einsum("...lhw,lr -> ...rhw", samples["samples"].cpu().float(), latent_rgb_factors)
        rgb = (((rgb + 1) / 2).clamp(0, 1))  # Change scale from -1..1 to 0..1
        rgb = rgb.movedim(1,-1)

        return (rgb,)


NODE_CLASS_MAPPINGS = {
    "LatentToRGB": LatentToRGB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentToRGB": "Latent to RGB",
}
