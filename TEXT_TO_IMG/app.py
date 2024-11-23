import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import tkinter as tko
import customtkinter as ctkob
from PIL import Image, ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
app = tko.Tk()
app.geometry("532x632")
app.title("Stable bud")
ctkob.set_appearance_mode("dark")

# Initialize the CTkEntry widget with the master argument
prompt = ctkob.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white")
prompt.place(x=10, y=10)

# Initialize the CTkLabel widget with the master argument
lmain = ctkob.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    image.save('generatedimage.png')
    img = Image.open('generatedimage.png')
    img = ImageTk.PhotoImage(img)
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to avoid garbage collection

# Initialize the CTkButton widget with the master argument
trigger = ctkob.CTkButton(master=app, height=40, width=120, text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
