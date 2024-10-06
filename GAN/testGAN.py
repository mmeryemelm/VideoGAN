import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import glob

from glob import glob
import cv2
import os
import numpy as np
from utils import process_and_write_video, process_and_write_image
from argparse import ArgumentParser

# Setup argument parser

parser = ArgumentParser()
parser.add_argument("-d", help="The dimension of each video, must be of shape [3,32,64,64]",nargs='*', default=[3,32,64,64])
parser.add_argument("-zd", help="The dimension of latent vector [100]", type=int, default=100)
parser.add_argument("-nb", help="The size of batch images [5]",type=int, default=5)
parser.add_argument("-c", help="The checkpoint file name",type=str, default="G_epoch_400")


args = parser.parse_args()

gesture_dict = {"gesture001": 0, "gesture002": 1, "gesture003": 2, "gesture004": 3, "gesture005": 4,
                "gesture006": 5, "gesture007": 6, "gesture008": 7, "gesture009": 8, "gesture010": 9,
                "gesture011": 10, "gesture012": 11, "gesture013": 12, "gesture014": 13, "gesture015": 14,
                "gesture016": 15, "gesture017": 16, "gesture018": 17, "gesture019": 18, "gesture020": 19,
                "gesture021": 20, "gesture022": 21, "gesture023": 22, "gesture024": 23}


class Generator(nn.Module):
    def __init__(self, zdim=args.zd, num_labels=24, embed_size=50):
        super(Generator, self).__init__()
        self.zdim = zdim
        self.label_embedding = nn.Embedding(num_labels, embed_size)


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






def load_frame1():
    data = []
    paths = glob("testdb/*/*/*")
    for path in paths:
        img = path + "/img_0000.jpg"
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 127.5 - 1
        data.append(frame)
    data = np.stack(data)
    print(data.shape)
    return torch.from_numpy(data).float().permute(0, 3, 1, 2), paths

def load_frame():
    data = []
    paths = glob("testdb/*/*/*")
    for path in paths:
        imgs = glob(path + "/frame_000.jpg")
        if imgs:
            img = imgs[0]
            frame = cv2.imread(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = (frame / 127.5) - 1
            data.append(frame)
    data = np.stack(data)
    print(data.shape)
    return torch.from_numpy(data).float().permute(0, 3, 1, 2), paths



def save_frames(video_tensor, frame_dir, gesture):
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    video_tensor = video_tensor.detach()
    video_tensor = video_tensor.squeeze(0)
    num_frames = video_tensor.size(1)
    for i in range(num_frames):
        frame = video_tensor[:, i, :, :]
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        frame_np = ((frame_np + 1) * 0.5 * 255).astype(np.uint8)
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(frame_dir, f"{gesture}_frame_{i:04d}.png"), frame_np)

def main():
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./genvideos"):
        os.makedirs("./genvideos")
    G = Generator(zdim=args.zd)
    if args.c:
        model_path = f"./checkpoints/{args.c}.pth"
        G.load_state_dict(torch.load(model_path), strict=True)
        print("Model restored from:", model_path)

    # Charger toutes les frames depuis la base de données
    frame_tensor, paths = load_frame()

    # Boucler sur tous les gestes et sujets
    for i, path in enumerate(paths):
        parts = path.split('\\')  # Adjust based on OS (use '/' for Unix systems)
        subject_name = parts[-3]  # Extract subject from the path
        gesture_name = parts[-2]  # Extract gesture from the path
        frame_dir = f"./genvideos/{subject_name}/{gesture_name}/{parts[-1]}"

        # Obtenir l'étiquette du geste à partir du dictionnaire
        if gesture_name in gesture_dict:
            chosen_label = torch.tensor([gesture_dict[gesture_name]]).long()

            # Générer une vidéo pour le geste et le sujet actuels
            gen_videos, _, _, _ = G(frame_tensor[i:i + 1], chosen_label)
            gen_videos = gen_videos.squeeze(0).detach()  # Remove batch dimension and detach

            video_np = gen_videos.cpu().numpy()  # Ensure it's on CPU and detached
            video_path = f"{frame_dir}/test_{gesture_name}.mp4"
            print(f"Generating video {video_path} with shape {video_np.shape}")

            save_frames(gen_videos, frame_dir, gesture_name)  # Save individual frames
            process_and_write_video(video_np, video_path)  # Optionally save complete video
        else:
            print(f"Gesture {gesture_name} not found in gesture dictionary.")

    print("Generated videos for all subjects and gestures.")

if __name__ == '__main__':
    main()


