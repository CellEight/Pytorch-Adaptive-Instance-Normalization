import sys
import torch
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms

def loadImage(filename):
    input_image = Image.open(filename).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0)

if __name__ == "__main__":
    # declare variables to store hooks even though they aren't nessary for inference you need them to load the model from file
    style_layers = ['1','6','10','20']
    debug_layers = [0,3,5,7]
    activations = [None]*4
    debug_activations = [None]*4
    debug_grads = [None]*4
    # declare hook function
    def styleHook(i, module, input, output):
        global activations
        activations[i] = output

    def debugHook(i, module, input, output):
        global activations
        debug_activations[i] = output

    content_img = loadImage(sys.argv[1])
    style_img = loadImage(sys.argv[2])

    model = torch.load('adain_model')

    output_img = model(content_img, style_img)
    save_image(output_img, 'output.jpg')
