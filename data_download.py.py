import spacy
from torchvision.models import resnet50, resnet101, resnet152
from torchvision.models import vgg16, vgg19
from torchvision.models import alexnet, squeezenet1_0, squeezenet1_1
from torchvision.models import densenet121, densenet169, densenet201
from torchvision.models import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
from torchvision.models import efficientnet_b6, efficientnet_b7
from torchvision.models import googlenet, inception_v3, resnext50_32x4d
from torchvision.models import resnext101_32x8d, wide_resnet50_2
from torchvision.models import wide_resnet101_2
from torchvision.models import shufflenet_v2_x1_5, shufflenet_v2_x2_0
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from torchvision.models import swin_b, swin_t
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
from nltk import download as nltk_download
import whisper

# Spacy
def download_spacy_models():
    try:
        spacy.cli.download("en_core_web_sm")
        print("Spacy model 'en_core_web_sm' downloaded successfully.")
        spacy.cli.download("en_core_web_md")
        print("Spacy model 'en_core_web_md' downloaded successfully.")
        spacy.cli.download("en_core_web_lg")
        print("Spacy model 'en_core_web_lg' downloaded successfully.")
        spacy.cli.download("en_core_web_trf")
        print("Spacy model 'en_core_web_trf' downloaded successfully.")
    except spacy.cli.Error as e:
        print(f"An error occurred while downloading the Spacy model: {e}")
    except Exception as e:
        print(f"An error occurred while downloading the Spacy model: {e}")

# NLTK
def download_nltk_resources():
    try:
        nltk_download("all")
    except Exception as e:
        print(f"An error occurred while downloading the NLTK resources: {e}")


# torchvision models
def download_torchvision_models():
    models = [
        # resnet50, resnet101, resnet152,
        # vgg16, vgg19,
        # alexnet, squeezenet1_0, squeezenet1_1,
        # densenet121, densenet169, densenet201,
        # mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small,
        # efficientnet_b6, efficientnet_b7,
        # googlenet, inception_v3,
        # resnext101_32x8d,
        # wide_resnet50_2, wide_resnet101_2,
        # shufflenet_v2_x1_5, shufflenet_v2_x2_0,
        # vit_b_16, vit_b_32, vit_l_16, 
        swin_b, swin_t,
        convnext_tiny, convnext_small, convnext_base, convnext_large,
        resnext50_32x4d
    ]
    
    for model in models:
        try:
            model(pretrained=True)
            print(f"{model.__name__} downloaded successfully.")
        except Exception as e:
            print(f"An error occurred while downloading {model.__name__}: {e}")

# Whisper models
def download_whisper_models():
    try:
        whisper.load_model("base")
        print("Whisper model 'base' downloaded successfully.")
        whisper.load_model("small")
        print("Whisper model 'small' downloaded successfully.")
        whisper.load_model("medium")
        print("Whisper model 'medium' downloaded successfully.")
        whisper.load_model("tiny")
        print("Whisper model 'tiny' downloaded successfully.")
    except Exception as e:
        print(f"An error occurred while downloading the Whisper models: {e}")


if __name__ == "__main__":
    # download_spacy_models()
    # download_nltk_resources()
    # download_torchvision_models()
    download_whisper_models()
    print("All models downloaded successfully.")
else:
    print("This script is intended to be run as a standalone program.")
    print("Please run it directly to download the torchvision models.")
    print("If you are importing this script, please call the download_torchvision_models function.")
    print("Exiting the script.")
    exit(0)