import torch
import torch.nn as nn
import os

class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class BrainTumorModel3d(nn.Module):
    """
    3D U-Net architecture to match 5D weights (N, C, D, H, W).
    """
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        self.encoder1 = ConvBlock3d(in_channels, 32) # Adjusted to 32 based on log mismatch
        self.encoder2 = ConvBlock3d(32, 64)
        self.encoder3 = ConvBlock3d(64, 128)
        
        self.pool = nn.MaxPool3d(2)
        
        self.bottleneck = ConvBlock3d(128, 256)
        
        self.up1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.decoder1 = ConvBlock3d(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.decoder2 = ConvBlock3d(128, 64)
        
        self.up3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.decoder3 = ConvBlock3d(64, 32)
        
        self.current_head = nn.Conv3d(32, out_channels, 1)
        self.future_head = nn.Conv3d(32, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (1, 4, D, H, W)
        s1 = self.encoder1(x)
        p1 = self.pool(s1)
        
        s2 = self.encoder2(p1)
        p2 = self.pool(s2)
        
        s3 = self.encoder3(p2)
        p3 = self.pool(s3)
        
        b = self.bottleneck(p3)
        
        d1_up = self.up1(b)
        d1 = self.decoder1(torch.cat([d1_up, s3], dim=1))
        
        d2_up = self.up2(d1)
        d2 = self.decoder2(torch.cat([d2_up, s2], dim=1))
        
        d3_up = self.up3(d2)
        d3 = self.decoder3(torch.cat([d3_up, s1], dim=1))
        
        curr = self.sigmoid(self.current_head(d3))
        fut = self.sigmoid(self.future_head(d3))
        
        return torch.cat([curr, fut], dim=1)

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Based on the size mismatch error, the base channel count seems to be 32
    model = BrainTumorModel3d(in_channels=4, out_channels=1)
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            # Handle potential DataParallel wrapping
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded 3D model weights.")
        except Exception as e:
            print(f"3D Model weight load error: {e}")
    
    model.to(device)
    model.eval()
    return model, device
