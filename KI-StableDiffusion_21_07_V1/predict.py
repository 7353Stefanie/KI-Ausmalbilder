import uuid
import os
from typing import List
from pathlib import Path

import torch
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

# Basispfad zu deinen LoRA-Dateien
BASE_LORA_PATH = "./modell"

animal_map = {
    "fox": "Fuchs",
    "fuchs": "Fuchs",
    "owl": "Eule",
    "eule": "Eule",
    "cat": "Katze",
    "katze": "Katze",
    "dog": "Hund",
    "hund": "Hund",
    "penguin": "Pinguin",
    "pinguin": "Pinguin",
    "rabbit": "Hase",
    "hase": "Hase",
    "deer": "Reh",
    "reh": "Reh",
    "bear": "Bär",
    "bär": "Bär",
}


MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_ID2 = "runwayml/stable-diffusion-v1-5"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"

def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

def generate_random_filename():
    short_id = uuid.uuid4()  # nur die ersten 8 Zeichen
    return f"Hase{short_id}.png"

def detect_animal_folder(prompt):
    prompt = prompt.lower()
    for en, de in animal_map.items():
        if en in prompt:
            return de
    return None


# Funktion: LoRA-Dateipfad erstellen
def get_lora_path_from_prompt(prompt):
    animal_folder = detect_animal_folder(prompt)
    if not animal_folder:
        return None  # kein passender Ordner
    full_path = os.path.join(BASE_LORA_PATH, animal_folder, "pytorch_lora_weights.safetensors")
    if os.path.isfile(full_path):
        return full_path
    else:
        print(f"⚠️ LoRA-Datei nicht gefunden für Tier: {animal_folder}")
        return None

def generate_images(
    prompt: str,
    negative_prompt: str = None,
    width: int = 768,
    height: int = 768,
    num_outputs: int = 1,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    scheduler_name: str = "DPMSolverMultistep",
    seed: int = None,
    output_dir: str = "/mnt/output_dir"
) -> List[Path]:
    print("CUDA verfügbar?", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("❌ CUDA NICHT verfügbar – läuft auf CPU!")

    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    if width * height > 1024 * 768:
        raise ValueError("Max image size is 1024x768 or 768x1024.")

    print("Loading safety checker...")
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        SAFETY_MODEL_ID,
        cache_dir=MODEL_CACHE,
        local_files_only=False,
    ).to("cuda")

    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE,
        safety_checker=safety_checker,
        local_files_only=False,
        torch_dtype=torch.float16,
    ).to("cuda")

# ---------------------------------------------------------------------------------------



    # Wichtig: Hier Lora-Weights korrekt laden
#    try:
# Jetzt beide Adapter kombinieren:

       # pipe.load_lora_weights({
       #     "hase1": "./train/lora-out/lora-out/neuer_Versuch_Lora_out/Hase/pytorch_lora_weights.safetensors",
       #     "hase2": "./train/lora-out/lora-out/neuer_Versuch_Lora_out/Hase2/pytorch_lora_weights.safetensors"
       # }, weights=[0.5, 0.5])



        # pipe.load_lora_weights("./train/lora-out/neuer_Versuch_Lora_out/Hase/pytorch_lora_weights.safetensors")

       # pipe.set_adapters(["hase1", "hase2"])
                      #         "hase1": "./train/lora-out/Hase1/pytorch_lora_weights.safetensors"
                      #        }, prefix=None)

 #   except Exception as e:
  #      print(f"⚠️ Konnte LoRA-Gewichte nicht laden: {e}")
    lora_path = get_lora_path_from_prompt(prompt)

    if lora_path:
        print(f"✅ LoRA-Pfad gefunden: {lora_path}"+lora_path)
        pipe.load_lora_weights(lora_path)  # ← Hier einbinden
    else:
        print("❌ Kein gültiger LoRA-Pfad gefunden. {lora_path}")


    pipe.scheduler = make_scheduler(scheduler_name, pipe.scheduler.config)
    generator = torch.Generator("cuda").manual_seed(seed)

    print("Generating image(s)...")
    output = pipe(
        prompt=[prompt] * num_outputs,
        negative_prompt=[negative_prompt] * num_outputs if negative_prompt else None,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=num_inference_steps,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    for i, image in enumerate(output.images):
        if hasattr(output, "nsfw_content_detected") and output.nsfw_content_detected[i]:
           print(f"NSFW content detected for image {i}, skipping.")
           continue
        filename = generate_random_filename()
        image_path = os.path.join(output_dir, filename)
        image.save(image_path)
        output_paths.append(Path(image_path))
        print(f"Bild gespeichert: {image_path}")

    if not output_paths:
        raise Exception("All outputs flagged as NSFW.")

    print(f"Saved {len(output_paths)} image(s): {[str(p) for p in output_paths]}")
    return output_paths

if __name__ == "__main__":
    print("Dies ist ein Hilfsmodul. Bitte 'run.py' starten.")

