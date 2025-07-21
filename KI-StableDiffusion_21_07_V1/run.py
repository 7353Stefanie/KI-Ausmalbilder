from predict import generate_images

if __name__ == "__main__":
    prompt = input("📝 Prompt eingeben: ")
    negative_prompt = input("🚫 Negative Prompt (leer lassen für keinen): ") or None

    generate_images(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=768,
        height=768,
        num_outputs=1,
        scheduler_name="DPMSolverMultistep",
    )
