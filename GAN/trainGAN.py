
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import os
import torchvision
import PIL
from utils import *
import numpy as np
from glob import glob
from random import shuffle
from datetime import datetime
from torchvision.models import VGG19_Weights
from argparse import ArgumentParser
import torchvision.models as models
import logging

import re
from torch.nn.utils import spectral_norm


parser = ArgumentParser()
parser.add_argument("-e", help="Epoch to train", type=int, default=400)
parser.add_argument("-d", help="The dimension of each video, must be of shape [3,32,64,64]", nargs='*', default=[3,32,64,64])
parser.add_argument("-zd", help="The dimension of latent vector [200]", type=int, default=100)
parser.add_argument("-nb", help="The size of batch images [8]", type=int, default=8)
parser.add_argument("-l", help="The value of sparsity regularizer [50]", type=float, default=50)
parser.add_argument("-c", help="The checkpoint epoch number (e.g., 1, 2, 3...)", type=int, default=None)
parser.add_argument("-s", help="Saving checkpoint file, every [1] epochs", type=int, default=1)
args = parser.parse_args()



"""
Initialisation de la configuration de journalisation pour le suivi des événements pendant l'entraînement.
Les messages de log seront écrits dans un fichier texte ainsi qu'affichés dans la console.
"""

log_filename = 'training_log.txt'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])


"""
Nous avons défini et essayé deux classes d'extracteurs de caractéristiques pour l'analyse vidéo et d'images.

1. **VideoFeatureExtractor** : Utilise le modèle préentraîné ResNet3D-18 (r3d_18) pour extraire des caractéristiques à partir de vidéos. 
2. **VGGFeatureExtractor** : Utilise le modèle VGG19 préentraîné pour extraire des caractéristiques d'images jusqu'à une certaine couche spécifiée (par défaut, la 9e couche).
"""


class VideoFeatureExtractor(nn.Module):
    def __init__(self):
        super(VideoFeatureExtractor, self).__init__()
        self.extractor = torchvision.models.video.r3d_18(pretrained=True)
        self.extractor.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.extractor(x)

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=9):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer_index])

    def forward(self, x):
        return self.feature_extractor(x)



class Generator(nn.Module):
    def __init__(self, zdim=args.zd, num_labels=24, embed_size=50):
        super(Generator, self).__init__()
        self.zdim = zdim
        self.label_embedding = nn.Embedding(num_labels, embed_size)

        """
        L'encodeur prend une image vidéo en entrée et la traite à travers plusieurs couches convolutives, 
        chacune suivie de normalisation de lot (BatchNorm) et d'une fonction d'activation ReLU. Chaque 
        couche extrait des caractéristiques de plus en plus abstraites à mesure que les dimensions spatiales 
        de l'entrée diminuent. La dernière couche d'encodage produit un tenseur de caractéristiques de taille 
        [zdim], qui est ensuite utilisé pour la génération de la séquence vidéo.
        """

        self.conv1a = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.dropout1a = nn.Dropout2d(0.35)  # Adding dropout
        self.conv2a = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.dropout2a = nn.Dropout2d(0.35)  # Adding dropout
        self.conv3a = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.dropout3a = nn.Dropout2d(0.35)  # Adding dropout
        self.conv4a = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4a = nn.BatchNorm2d(512)
        self.dropout4a = nn.Dropout2d(0.35)  # Adding dropout
        self.conv5a = nn.Conv2d(512, zdim, kernel_size=4, stride=2, padding=1)
        self.bn5a = nn.BatchNorm2d(zdim)  # Added batch norm here too
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        """
        Le décodeur prend le vecteur de caractéristiques généré par l'encodeur, concaténé avec du bruit 
        aléatoire et l'embedding des étiquettes, et le traite via des couches de déconvoution 
        (ConvTranspose2d (arrière-plan) et ConvTranspose3d (premier-plan)). 
        Le décodeur reconstruit progressivement une vidéo en augmentant la taille spatiale des données. 
        Des couches de normalisation et de fonctions d'activation (ReLU, tanh) sont appliquées à chaque étape de la génération.
        Le masque contrôle la combinaison entre les deux flux. (arrière plan et premier plan)
        """


        self.conv1b = nn.ConvTranspose2d(zdim+zdim+embed_size, 512, [3,3], [1,1], [0,0])
        self.bn1b = nn.BatchNorm2d(512)
        self.conv2b = nn.ConvTranspose2d(512, 256, [4,4], [2,2], [1,1])
        self.bn2b = nn.BatchNorm2d(256)
        self.conv3b = nn.ConvTranspose2d(256, 128, [4,4], [2,2], [1,1])
        self.bn3b = nn.BatchNorm2d(128)
        self.conv4b = nn.ConvTranspose2d(128, 64, [4,4], [2,2], [1,1])
        self.bn4b = nn.BatchNorm2d(64)
        self.conv5b = nn.ConvTranspose2d(64, 3, [4,4], [2,2], [1,1])

        self.convfg1 = nn.ConvTranspose3d(zdim+zdim+embed_size, 512, [2, 3, 3], [1, 1, 1], [0, 0, 0])
        self.bnfg1 = nn.BatchNorm3d(512)
        self.convfg2 = nn.ConvTranspose3d(512, 256, [4,4,4], [2,2,2], [1,1,1])
        self.bnfg2 = nn.BatchNorm3d(256)
        self.convfg3 = nn.ConvTranspose3d(256, 128, [4,4,4], [2,2,2], [1,1,1])
        self.bnfg3 = nn.BatchNorm3d(128)
        self.convfg4 = nn.ConvTranspose3d(128, 64, [4,4,4], [2,2,2], [1,1,1])
        self.bnfg4 = nn.BatchNorm3d(64)
        self.convfg5 = nn.ConvTranspose3d(64, 3, [4,4,4], [2,2,2], [1,1,1])
        self.conv5m = nn.ConvTranspose3d(64, 1, [4,4,4], [2,2,2], [1,1,1])

        self._initialize_weights()

    """
    Le `forward` applique le traitement de l'encodeur aux données d'entrée, puis génère un vecteur de 
    caractéristiques `z`. Ensuite, il génère un bruit aléatoire de même dimension que `z`, concatène le 
    bruit avec `z`, et ajoute l'embedding des étiquettes pour conditionner la génération. 
    Le décodeur prend ces caractéristiques concaténées pour générer une vidéo à l'aide d'un masque et des flux distincts.
    La vidéo finale est une combinaison pondérée des flux où le masque détermine la contribution de chacun à chaque pixel.
    """

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, labels):
        # Traitement initial pour obtenir 'z'
        x = self.dropout1a(self.relu(self.bn1a(self.conv1a(x))))
        x = self.dropout2a(self.relu(self.bn2a(self.conv2a(x))))
        x = self.dropout3a(self.relu(self.bn3a(self.conv3a(x))))
        x = self.dropout4a(self.relu(self.bn4a(self.conv4a(x))))
        z = self.bn5a(self.relu(self.conv5a(x)))

        # Générer du bruit adapté aux dimensions de 'z'
        noise = torch.randn(z.shape[0], self.zdim, z.size(2), z.size(3), device=z.device)

        # Concaténer le bruit avec 'z'
        z = torch.cat([z, noise], dim=1)

        # Ajout de l'embedding des étiquettes et concaténation
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(label_embedding.size(0), label_embedding.size(1), 1, 1)
        label_embedding = label_embedding.expand(-1, -1, z.size(2), z.size(3))
        z = torch.cat([z, label_embedding], dim=1)

        # Suite des opérations pour générer 'b'
        b = self.conv1b(z)
        b = self.bn1b(b)
        b = F.relu(self.bn2b(self.conv2b(b)))
        b = F.relu(self.bn3b(self.conv3b(b)))
        b = F.relu(self.bn4b(self.conv4b(b)))
        b = torch.tanh(self.conv5b(b)).unsqueeze(2)

        z = z.unsqueeze(2)
        f = F.relu(self.bnfg1(self.convfg1(z)))
        f = F.relu(self.bnfg2(self.convfg2(f)))
        f = F.relu(self.bnfg3(self.convfg3(f)))
        f = F.relu(self.bnfg4(self.convfg4(f)))
        m = torch.sigmoid(self.conv5m(f))
        f = torch.tanh(self.convfg5(f))

        # Combinaison pour obtenir la sortie finale
        out = m * f + (1 - m) * b
        return out, f, b, m


class Discriminator(nn.Module):
    def __init__(self, num_labels=24, embed_size=50):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_labels, embed_size)

        """
        L'encodeur du discriminateur prend en entrée une vidéo 3D, concaténée avec un embedding des étiquettes, 
        et la passe à travers plusieurs couches convolutives 3D. Chaque couche réduit progressivement la taille 
        spatiale et temporelle de l'entrée tout en extrayant des caractéristiques pertinentes à chaque étape. 
        Les couches sont suivies d'une normalisation de lot (BatchNorm3d) et d'une fonction d'activation LeakyReLU 
        pour capturer des informations complexes, ainsi que de Dropout3D pour éviter le surapprentissage.
        """

        self.conv1 = nn.Conv3d(3 + embed_size, 64, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.dropout1 = nn.Dropout3d(0.35)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.dropout2 = nn.Dropout3d(0.35)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.dropout3 = nn.Dropout3d(0.35)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        self.dropout4 = nn.Dropout3d(0.35)
        self.conv5 = nn.Conv3d(512, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 2, 2))

        self._initialize_weights()

    """
    La méthode `forward` prend une séquence vidéo et un label, concatène l'embedding du label avec la vidéo, 
    et passe cette combinaison à travers plusieurs couches convolutives 3D. Ces couches réduisent progressivement 
    les dimensions spatiales et temporelles tout en capturant des informations. À chaque étape, les activations 
    sont transformées par la fonction LeakyReLU et la normalisation de lot, suivies d'un dropout pour aider à 
    la régularisation. à la fin du traitement, la sortie est une prédiction pour chaque vidéo, indiquant 
    si elle est réelle ou générée.
    """

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, y, labels):
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(label_embedding.size(0), label_embedding.size(1), 1, 1, 1)
        label_embedding = label_embedding.expand(-1, -1, y.size(2), y.size(3), y.size(4))

        y = torch.cat([y, label_embedding], dim=1)

        y = F.leaky_relu(self.bn1(self.conv1(y)), 0.2)
        y = self.dropout1(y)
        y = F.leaky_relu(self.bn2(self.conv2(y)), 0.2)
        y = self.dropout2(y)
        y = F.leaky_relu(self.bn3(self.conv3(y)), 0.2)
        y = self.dropout3(y)
        y = F.leaky_relu(self.bn4(self.conv4(y)), 0.2)
        y = self.dropout4(y)
        y = self.conv5(y)
        return y



"""
Cette fonction calcule le Gradient penalty pour renforcer les gradients du discriminateur et stabiliser l'apprentissage.
Nous avons décidé de ne pas l'appliquer
"""
def gradient_penalty(D, real_data, fake_data, real_labels):
    alpha = torch.rand(real_data.size(0), 1, 1, 1, 1).to(real_data.device)  # For 3D data
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)
    disc_interpolates = D(interpolates, real_labels)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_pen = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_pen





"""
Cette fonction charge et prépare les données vidéo ainsi que leurs étiquettes pour l'entraînement du modèle. 
Elle parcours les chemins des fichiers à partir du répertoire pour trouver les vidéos où chacun est représentée par une séquence d'images.
Nous nous assurons que le nombre d'images est au maximum 32 frames, puis on les normalise en valeurs comprises entre [-1, 1].
Ces vidéos, leurs étiquettes ainsi que les IDs des sujets sont stockées dans des listes converties en tableaux NumPy pour être utilisables dans le modèle.
"""
def load_data():
    data = []
    labels = []
    subject_ids = []
    gesture_dict = {
        "gesture001": 0, "gesture002": 1, "gesture003": 2, "gesture004": 3, "gesture005": 4,
        "gesture006": 5, "gesture007": 6, "gesture008": 7, "gesture009": 8, "gesture010": 9,
        "gesture011": 10, "gesture012": 11, "gesture013": 12, "gesture014": 13, "gesture015": 14,
        "gesture016": 15, "gesture017": 16, "gesture018": 17, "gesture019": 18, "gesture020": 19,
        "gesture021": 20, "gesture022": 21, "gesture023": 22, "gesture024": 23
    }
    paths = glob("D:\\finalgan\\try\\*\\*\\*")
    max_frames = 32
    print("Total directories found:", len(paths))
    for path in paths:
        video = []
        imgs = glob(path + "\\*.jpg")
        gesture_label = path.split(os.sep)[-2]
        subject_id = path.split(os.sep)[-3]
        if gesture_label not in gesture_dict:
            print("Skipping unrecognized gesture:", gesture_label)
            continue
        if not imgs:
            continue
        interval = max(len(imgs) // max_frames, 1)
        sampled_imgs = imgs[::interval][:max_frames]
        for img in sampled_imgs:
            frame = cv2.imread(img)
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 127.5 - 1  # Normalize
            video.append(frame)
        if len(video) < max_frames:
            last_frame = video[-1]
            video.extend([last_frame] * (max_frames - len(video)))
        video = np.stack(video)
        data.append(video)
        labels.append(gesture_dict[gesture_label])
        subject_ids.append(int(subject_id.replace('subject', '')))
    data = np.array(data)
    labels = np.array(labels)
    subject_ids = np.array(subject_ids)
    print(f"Loaded {len(data)} videos with labels and subject IDs.")
    return data, labels, subject_ids

"""
Cette fonction enregistre les frames d'une vidéo générée dans un dossier spécifique à l'itération (epoch) et à l'étiquette (label).
On inverse la normalisation pour retrouver les valeurs de spixels de [0, 255] puis on converti du format RGB au format BGR pour être compatible avec la fonction `cv2.imwrite`
"""
def save_frames(video_frames, epoch, label_name, subject_id, videos_folder):
    video_folder = os.path.join(videos_folder, f"g{label_name}s{subject_id:02d}_epoch_{epoch+1}")
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    for frame_idx, frame in enumerate(video_frames):
        frame = ((frame * 0.5 + 0.5) * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_path = os.path.join(video_folder, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_path, frame)







"""
Cette fonction déforme une image (frame) en utilisant un champ de flux optique. Le flux optique fournit des 
vecteurs de déplacement qui indiquent comment chaque pixel doit être déplacé d'une image à l'autre.
"""
def warp_flow(frame, flow):
        if len(frame.size()) == 3:
        frame = frame.unsqueeze(0)
    B, C, H, W = frame.size()
    grid_x, grid_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H),indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=0).float().to(flow.device)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    if flow.size(1) != 2:
        flow = flow.permute(0, 3, 1, 2)
    warped_grid = grid + flow
    warped_grid[:, 0, :, :] = 2.0 * warped_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
    warped_grid[:, 1, :, :] = 2.0 * warped_grid[:, 1, :, :] / max(H - 1, 1) - 1.0
    warped_grid = warped_grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
    warped_frame = F.grid_sample(frame, warped_grid, align_corners=True)
    return warped_frame

"""
Cette fonction calcule la perte basée sur le flux optique en déformant les frames générées en utilisant le flux optique des 
frames réelles, puis en comparant les résultats.
"""
def warp_optical_flow_loss(real_video, generated_video):
    batch_size = real_video.size(0)
    total_loss = 0.0
    for i in range(batch_size):
        for t in range(real_video.size(2) - 1):
            real_flow = calculate_optical_flow(real_video[i, :, t], real_video[i, :, t + 1])
            real_flow_tensor = torch.tensor(real_flow, dtype=torch.float32).unsqueeze(0).to(real_video.device)
            warped_gen_frame = warp_flow(generated_video[i, :, t].unsqueeze(0), real_flow_tensor)
            real_next_frame = real_video[i, :, t + 1].unsqueeze(0)
            total_loss += F.l1_loss(warped_gen_frame, real_next_frame)
    return total_loss / batch_size

"""
Cette fonction calcule le flux optique entre deux frames successives à l'aide de la méthode de Farneback.
"""
def calculate_optical_flow(prev_frame, next_frame):
    prev_frame_np = (prev_frame.squeeze().permute(1, 2, 0).cpu().detach().numpy() + 1) * 0.5 * 255
    next_frame_np = (next_frame.squeeze().permute(1, 2, 0).cpu().detach().numpy() + 1) * 0.5 * 255
    prev_gray = cv2.cvtColor(prev_frame_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_frame_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


"""
Cette fonction calcule la perte d'interpolation de frames en comparant les frames générées à des frames réelles interpolées.
"""
def frame_interpolation_loss(real_video, generated_video):
        batch_size = real_video.size(0)
    total_loss = 0.0
    for i in range(batch_size):
        for t in range(real_video.size(2) - 1):
            interpolated_real_frame = (real_video[i, :, t] + real_video[i, :, t + 1]) / 2.0
            total_loss += F.l1_loss(generated_video[i, :, t + 1], interpolated_real_frame)
    return total_loss / batch_size

"""
Cette fonction calcule la perte de cohérence temporelle en mesurant la différence entre les variations entre frames 
réelles et générées."""
def temporal_coherence_loss(real_video, generated_video):
    coherence_loss = 0
    for t in range(real_video.size(2) - 1):
        real_diff = real_video[:, :, t + 1] - real_video[:, :, t]
        gen_diff = generated_video[:, :, t + 1] - generated_video[:, :, t]
        coherence_loss += F.l1_loss(real_diff, gen_diff)
    return coherence_loss



"""
Cette fonction calcule la perte de flux optique en comparant les flux optiques calculés à partir des vidéos réelles et générées.
"""
def optical_flow_loss(real_video, generated_video):
    flow_loss = 0
    batch_size = real_video.size(0)
    total_frames = 0
    for i in range(batch_size):
        for t in range(real_video.size(2) - 1):
            real_flow = calculate_optical_flow(real_video[i, :, t], real_video[i, :, t + 1])
            gen_flow = calculate_optical_flow(generated_video[i, :, t], generated_video[i, :, t + 1])
            flow_loss += F.mse_loss(torch.tensor(real_flow), torch.tensor(gen_flow), reduction='sum')
            total_frames += 1
            logging.debug(f"Batch {i}, Frame {t}, Real flow mean: {real_flow.mean()}, Gen flow mean: {gen_flow.mean()}")
    if total_frames > 0:
        flow_loss /= total_frames
        logging.debug(f"Total optical flow loss: {flow_loss.item()} over {total_frames} frames")
    else:
        flow_loss = torch.tensor(0.0)
        logging.debug("No frames were compared.")
    return flow_loss


"""
Cette fonction trouve le dernier checkpoint enregistré pour le générateur (G) dans un répertoire de checkpoints.
"""
def find_latest_checkpoint():
    checkpoint_files = glob("./checkpoints/G_epoch_*.pth")
    if not checkpoint_files:
        return None
    checkpoint_epochs = [int(re.search(r'G_epoch_(\d+).pth', f).group(1)) for f in checkpoint_files]
    latest_epoch = max(checkpoint_epochs)
    return latest_epoch



def main():
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./genvideos"):
        os.makedirs("./genvideos")


        """On défini un dictionnaire `gesture_dict` qui mappe les noms des gestes à des index numériques.
       Le dictionnaire inverse `reverse_gesture_dict` est créé pour convertir les index en noms de gestes au moment de la génération des vidéos."""

    gesture_dict = {"gesture001": 0, "gesture002": 1, "gesture003": 2, "gesture004": 3, "gesture005": 4,
                    "gesture006": 5, "gesture007": 6, "gesture008": 7, "gesture009": 8, "gesture010": 9,
                    "gesture011": 10, "gesture012": 11, "gesture013": 12, "gesture014": 13, "gesture015": 14,
                    "gesture016": 15, "gesture017": 16, "gesture018": 17, "gesture019": 18, "gesture020": 19,
                    "gesture021": 20, "gesture022": 21, "gesture023": 22, "gesture024": 23}
    reverse_gesture_dict = {v: k for k, v in gesture_dict.items()}

    videos_folder = "./generatedtrain"
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #feature_extractor = VideoFeatureExtractor().to(device)

    G = Generator(zdim=args.zd).to(device)
    D = Discriminator().to(device)
    start_epoch = 0
    if args.c is None:
        args.c = find_latest_checkpoint()

    if args.c is not None:
        print(f"Attempting to load checkpoint for epoch: {args.c}")
        try:
            G_checkpoint_path = f"./checkpoints/G_epoch_{args.c}.pth"
            D_checkpoint_path = f"./checkpoints/D_epoch_{args.c}.pth"
            G.load_state_dict(torch.load(G_checkpoint_path, map_location=device, weights_only=True), strict=True)
            D.load_state_dict(torch.load(D_checkpoint_path, map_location=device, weights_only=True), strict=True)
            start_epoch = args.c
            print(f"Model restored from checkpoint: Epoch {args.c}")
        except FileNotFoundError:
            print(f"Checkpoint not found for Epoch {args.c}. Starting training from scratch.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    vgg = VGGFeatureExtractor().to(device)
    vgg.eval()


    G.to(device)
    D.to(device)
    weight_decay = 1e-5
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=weight_decay)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))



    data, labels, subject_ids = load_data()
    data_labels = list(zip(data, labels, subject_ids))
    shuffle(data_labels)
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    for epoch in range(start_epoch, args.e + 1):

        shuffle(data_labels)

        """
        Pour chaque batch de données, les vidéos et les labels réels sont extraits du dataset. La taille de batch est `args.nb`.
        Les vidéos réelles sont converties en tenseur PyTorch et leur format est réorganisé (permute) pour correspondre aux dimensions attendues par le modèle (batch, channels, depth, height, width). Les labels réels sont également convertis en tenseur.
        Les premières images I des vidéos réelles sont extraites et transformées en tenseur. Ces premières frames serviront d'entrée pour le générateur.
        Le générateur prend ces premières frames ainsi que les labels réels comme entrées pour générer une vidéo, ainsi que les tenseurs f, b et m.
        L'optimiseur du discriminateur est réinitialisé.
        Les prédictions du discriminateur sont calculées pour les vidéos réelles (`logit_real`) et  générées (`logit_fake`). """

        for counter in range(int(len(data_labels) / args.nb)):
            batch = data_labels[counter * args.nb: (counter + 1) * args.nb]
            real_videos, real_labels, real_subject_ids = zip(*batch)

            real_video_tensor = torch.tensor(np.stack(real_videos), dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)
            real_labels_tensor = torch.tensor(real_labels, dtype=torch.long).to(device)


            first_frames = [video[0] for video in real_videos]
            first_frame_tensor = torch.tensor(np.array(first_frames), dtype=torch.float32).permute(0, 3, 1, 2).to(device)

            fake_video, f, b, m = G(first_frame_tensor, real_labels_tensor)

            optimizer_D.zero_grad()
            logit_real = D(real_video_tensor, real_labels_tensor)
            logit_fake = D(fake_video.detach(), real_labels_tensor)

            prob_real = torch.mean(torch.sigmoid(logit_real))
            prob_fake = torch.mean(torch.sigmoid(logit_fake))

            """
            La perte pour les prédictions sur les vidéos réelles (`loss_D_real`) est calculée en utilisant la fonction `binary_cross_entropy_with_logits`. 
            Cette fonction compare les prédictions du discriminateur sur les vidéos réelles avec des étiquettes de `1` (c'est-à-dire que les vidéos réelles doivent être classées comme réelles).
            La perte pour les prédictions sur les vidéos générées (`loss_D_fake`) est calculée en comparant les prédictions du discriminateur sur les vidéos générées avec des étiquettes de `0` (les vidéos générées doivent être classées comme fausses).
            La perte totale du discriminateur (`loss_D`) est la moyenne des deux pertes calculées."""

            loss_D_real = F.binary_cross_entropy_with_logits(logit_real, torch.ones_like(logit_real))
            loss_D_fake = F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2

            real_preds = D(real_video_tensor, real_labels_tensor)
            fake_preds = D(fake_video.detach(), real_labels_tensor)
            d_loss = adversarial_loss_discriminator(real_preds, fake_preds)
            d_loss.backward()

            loss_D.backward()
            optimizer_D.step()

            """
            L'entraînement du générateur est effectué après chaque 5 itérations du discriminateur.
            Les prédictions du discriminateur pour les vidéos générées (`logit_gen`) sont calculées  
            la perte adversariale du générateur (`loss_gen_adv`) est déterminée en comparant ces prédictions à des étiquettes de `1` (le générateur souhaite que le discriminateur classifie les vidéos générées comme réelles).
            Plusieurs autres pertes sont calculées pour améliorer la qualité des vidéos générées :
              **Perte L1 (`loss_gen_l1`)** : Mesure la différence absolue entre les vidéos réelles et générées.
              **Perte L2 (`loss_gen_l2`)** : Mesure la différence au carré entre les vidéos réelles et générées.
              **Perte VGG (`vgg_loss`)** : Calcule la différence entre les caractéristiques extraites des vidéos réelles et générées à l'aide du réseau VGG19, pour capturer des similarités perceptuelles.
            D'autres pertes sont testées pour améliorer le modèle :
              **Perte de cohérence temporelle (`coherence_loss`)** 
              **Perte de flux optique (`flow_loss`)** 
              **Perte d'interpolation de frames (`interp_loss`)**
              **Perte de caractéristique (`g_loss_feat`)**
            Différentes combinaisons de pondérations sont testées pour ajuster la contribution de chaque perte.
            """

            discriminator_iterations += 1
            if discriminator_iterations % 5 == 0:  # Every 5th iteration is generator's turn

                optimizer_G.zero_grad()
                logit_gen = D(fake_video, real_labels_tensor)
                loss_gen_adv = F.binary_cross_entropy_with_logits(logit_gen, torch.ones_like(logit_gen))
                loss_gen_l1 = l1_loss(fake_video, real_video_tensor)
                loss_gen_l2 = l2_loss(fake_video, real_video_tensor)
                real_features = vgg(real_video_tensor[:, :, 0, :, :])
                fake_features = vgg(fake_video[:, :, 0, :, :])
                vgg_loss = l2_loss(fake_features, real_features)


                ''' d'autres pertes qu'on a essayé pour améliorer notre modèle'''
                coherence_loss = temporal_coherence_loss(real_video_tensor, fake_video)
                flow_loss = warp_optical_flow_loss(real_video_tensor, fake_video)
                interp_loss = frame_interpolation_loss(real_video_tensor, fake_video)
                real_features = feature_extractor(real_video_tensor)
                fake_features = feature_extractor(fake_video)
                g_loss_feat = F.mse_loss(fake_features, real_features)

                ''' d'autres pertes qu'on a essayé pour améliorer notre modèle'''
                #loss_G = 0.1 * loss_gen_adv + loss_gen_l1 + 0.1 * loss_gen_l2 + 0.1 * vgg_loss + 1 * flow_loss + 0.5 * frame_loss + 0.5 * coherence_loss
                #loss_G = 0.1 * loss_gen_adv + 0.2 * loss_gen_l1 + 0.05 * loss_gen_l2 + 0.05 * vgg_loss + 0.05 * flow_loss + 0.05 * coherence_loss + 0.05 * interp_loss
                #loss_G = 0.5 * loss_gen_adv + 0.3 * loss_gen_l1 + 0.002 * loss_gen_l2 + 0.04 * vgg_loss

                loss_G = 0.5 * loss_gen_adv + 0.2 * loss_gen_l1 + 0.02 * loss_gen_l2 + 0.4 * vgg_loss


                loss_G.backward()
                #torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)  # Gradient clipping
                #torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)  # Gradient clipping

                optimizer_G.step()
                optimizer_D.step()


                print(
                    f"Epoch {epoch + 1}, Batch {counter + 1}, Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}, "
                    f"D(x): {prob_real:.4f}, D(G(z)): {prob_fake:.4f}, L1: {loss_gen_l1.item():.4f}, L2: {loss_gen_l2.item():.4f}, "
                    f"VGG: {vgg_loss.item():.4f}, Optical Flow: {flow_loss.item():.4f}, frame loss: {frame_loss.item():.4f}, coherenceloss: {coherence_loss.item():.4f}, interp loss: {interp_loss.item():.4f}, feature loss: {g_loss_feat.item():.4f} ")

                logging.info(
                    f"Epoch {epoch + 1}, Batch {counter + 1}, Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}, "
                    f"D(x): {prob_real:.4f}, D(G(z)): {prob_fake:.4f}, L1: {loss_gen_l1.item():.4f}, L2: {loss_gen_l2.item():.4f}, "
                    f"VGG: {vgg_loss.item():.4f}, Optical Flow: {flow_loss.item():.4f}, frame loss: {frame_loss.item():.4f}, coherenceloss: {coherence_loss.item():.4f}, interp loss: {interp_loss.item():.4f}, feature loss: {g_loss_feat.item():.4f} ")




            """
            À chaque epoch args.s le modèle génère et enregistre une vidéo et des images.
            Les checkpoints des poids du générateur (`G`) et du discriminateur (`D`) sont sauvegardés"""

            if (epoch + 1) % args.s == 0 and counter == int(len(data_labels) / args.nb) - 1:
                video_path = os.path.join(videos_folder, f"{epoch + 1}_batch_{counter + 1}.mp4")
                process_and_write_video(fake_video[0:1].cpu().data.numpy(), video_path)
                process_and_write_image(b.cpu().data.numpy(), video_path)

                # Ensure that `label_idx` is a scalar tensor or extract its scalar value before using it
                #for video_idx, (video_frames, label_idx) in enumerate(zip(fake_video, real_labels)):
                for video_idx, (video_frames, label_idx, subject_id) in enumerate(
                            zip(fake_video, real_labels, real_subject_ids)):
                    video_frames = video_frames.permute(1, 2, 3, 0).detach().cpu().numpy()
                    label_name = reverse_gesture_dict[label_idx]
                    save_frames(video_frames, epoch, label_name, subject_id, videos_folder)

                torch.save(G.state_dict(), f"./checkpoints/G_epoch_{epoch + 1}.pth")
                torch.save(D.state_dict(), f"./checkpoints/D_epoch_{epoch + 1}.pth")
                print(f"Checkpoints saved for epoch {epoch + 1}")



if __name__ == '__main__':
    main()
