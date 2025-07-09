import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# --- Model Architecture (Copied from net.py) ---

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class Generator_makeup(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, repeat_num=6, input_nc=6):
        super(Generator_makeup, self).__init__()

        layers = []
        layers.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers_1 = []
        layers_1.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_1.append(nn.Tanh())
        self.branch_1 = nn.Sequential(*layers_1)
        layers_2 = []
        layers_2.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_2.append(nn.Tanh())
        self.branch_2 = nn.Sequential(*layers_2)

    def forward(self, x, y):
        input_x = torch.cat((x, y), dim=1)
        out = self.main(input_x)
        out_A = self.branch_1(out)
        out_B = self.branch_2(out)
        return out_A, out_B

# --- Inference Logic ---

def load_model(model_path):
    """Loads the pre-trained generator model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Initialize model
    G = Generator_makeup()
    
    # Load state dict
    # Using map_location to load on CPU if CUDA is not available or model was trained on GPU
    G.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    G.eval() # Set model to evaluation mode
    return G

def preprocess_image(image_path, img_size=128):
    """Loads and preprocesses an image from a given path."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def deprocess_image(tensor):
    """Converts a tensor back to a PIL image."""
    # Reverse the normalization
    tensor = tensor.squeeze(0).cpu().clamp_(-1, 1) * 0.5 + 0.5
    return transforms.ToPILImage()(tensor)

def transfer_makeup(model, no_makeup_path, makeup_path, output_path):
    """
    Performs makeup transfer on a given pair of images.

    Args:
        model: The pre-trained BeautyGAN generator model.
        no_makeup_path (str): Path to the source (no-makeup) image.
        makeup_path (str): Path to the reference (makeup) image.
        output_path (str): Path to save the resulting image.
    """
    # Preprocess images
    org_img = preprocess_image(no_makeup_path)
    ref_img = preprocess_image(makeup_path)

    # Perform inference
    with torch.no_grad(): # No need to track gradients for inference
        result_img_tensor, _ = model(org_img, ref_img)

    # Postprocess and save the result
    result_img = deprocess_image(result_img_tensor)
    result_img.save(output_path)
    print(f"Result saved to {output_path}")

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # This is a placeholder for where the model file should be.
    # The user needs to provide this file.
    MODEL_PATH = 'D:/PythonProject/face-recognition-001/BeautyGAN-PyTorch-reimplementation-master/snapshot/200_G.pth'
    
    # Example image paths (assuming they exist)
    NO_MAKEUP_IMG = 'path/to/your/no_makeup_image.jpg'
    MAKEUP_IMG = 'path/to/a/makeup_style.png'
    OUTPUT_IMG = 'path/to/your/result.jpg'

    print("Starting makeup transfer...")
    print("Reminder: The model path needs to be provided by the user.")

    if not os.path.exists(MODEL_PATH):
        print("\n!!!---- MODEL FILE NOT FOUND ----!!!")
        print(f"Please place the pre-trained model (e.g., '200_G.pth') at: {MODEL_PATH}")
        print("Cannot proceed without the model file.")
    elif not os.path.exists(NO_MAKEUP_IMG) or not os.path.exists(MAKEUP_IMG):
        print("\n!!!---- INPUT IMAGES NOT FOUND ----!!!")
        print(f"Please provide valid paths for the no-makeup and makeup images.")
        print(f"- No Makeup Path: {NO_MAKEUP_IMG}")
        print(f"- Makeup Path: {MAKEUP_IMG}")
    else:
        try:
            # 1. Load Model
            print(f"Loading model from {MODEL_PATH}...")
            beauty_gan_model = load_model(MODEL_PATH)
            print("Model loaded successfully.")

            # 2. Perform Transfer
            print(f"Applying makeup from {MAKEUP_IMG} to {NO_MAKEUP_IMG}...")
            transfer_makeup(beauty_gan_model, NO_MAKEUP_IMG, MAKEUP_IMG, OUTPUT_IMG)
            print("\nMakeup transfer complete!")
            print(f"Result saved to: {OUTPUT_IMG}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")