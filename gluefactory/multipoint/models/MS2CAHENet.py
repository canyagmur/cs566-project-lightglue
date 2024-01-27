import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn

from gluefactory.multipoint.models.SRHENNet import HNet

from gluefactory.multipoint.utils.homographies import WarpingModule



def estimate_homography(displacement,img_size):

    h,w = img_size
    original_src_points = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
    original_src_points = torch.from_numpy(original_src_points).to(displacement.device) #TODO i need to think here
    #print(displacement)
    # Apply the displacement to the original destination points to get new destination points
    #displacement_scaled = displacement*32 * h/128
    new_dst_points = original_src_points + displacement
    #print(img_size,displacement)

    # Convert the 4-point parameterization to the homography matrix using DLT
    homography = [cv2.findHomography(original_src_points.cpu().detach().numpy(), new_dst_points[i].cpu().detach().numpy())[0]
                for i in range(displacement.shape[0])]
    homography = np.stack(homography, axis=0)
    homography = torch.from_numpy(homography).to(displacement.device)

    return homography




def warp_image_with_homography(target_image, homography):
    
    # Check if tensors require gradients
    target_image_req_grad = target_image.requires_grad
    homography_req_grad = homography.requires_grad
    
    # Set requires_grad = False for CV2 operations
    target_image = target_image.detach()  
    homography = homography.detach()
            
    device = target_image.device 
    target_image = target_image.cpu().numpy().transpose(0, 2, 3, 1) 
    homography = homography.cpu().numpy()

    # Apply the homography transformation to the target image using OpenCV
    warped_images = []
    for i in range(target_image.shape[0]):
        h, w = target_image[i].shape[:2]
        warped_image = cv2.warpPerspective(target_image[i], homography[i], (w, h))
        warped_images.append(warped_image)
    # Convert the warped image back to a PyTorch tensor
    warped_images = np.array(warped_images)
    warped_images = warped_images.transpose(0, 3, 1, 2)
    warped_images = torch.from_numpy(warped_images).to(device)
    
    # Set requires_grad back to original values
    if target_image_req_grad:
        warped_images.requires_grad_()
    if homography_req_grad:   
        homography.requires_grad_()
        
    return warped_images


class MyNetwork(nn.Module):
    def __init__(self,original_img_size=128):
        super(MyNetwork, self).__init__()



        self.original_img_size = original_img_size
        self.model = HNet() 
        self.warper = WarpingModule()

    def warp_image(self,image,homography):
        warped_image = self.warper(image.type(torch.float32), homography.type(torch.float32), image.shape[2:], 'bilinear', 'reflection')
        #print(image.requires_grad,warped_image.requires_grad) #look here
        return warped_image

    def prep_images(self,image3R,image3T):
        # Resize to the desired sizes
        image2R = F.interpolate(image3R, scale_factor=0.5, mode='bilinear', align_corners=False)
        image2T = F.interpolate(image3T, scale_factor=0.5, mode='bilinear', align_corners=False)
        image1R = F.interpolate(image2R, scale_factor=0.5, mode='bilinear', align_corners=False)
        image1T = F.interpolate(image2T, scale_factor=0.5, mode='bilinear', align_corners=False)

        return (image2R,image2T,image1R,image1T)

    # Stage 1: Estimate displacement D1 and homography matrix H1 between smallest-resolution images I1R and I1T.
    def stage1(self,image1R, image1T, model):
        #print("stage 1 input shape :",image1T.shape)
        displacement_output = model(x2=image1T,x1=image1R,stage=1)
        D1 = displacement_output.reshape((-1,4, 2))
        H1 = estimate_homography(D1,img_size=(32,32))

        return D1, H1

    # Stage 2: Estimate homography transformation between the reference image I2R and the warped target image bI2T.
    def stage2(self,image2R, image2T, model,H1,D1,scale):

        warped_image2T = self.warp_image(image2T, torch.inverse(H1*scale))
        displacement_output = model(x2=warped_image2T,x1=image2R,stage=2)
        displacement_output = displacement_output.reshape((-1,4, 2))
        D2 = D1*2 + displacement_output
        H2 = estimate_homography(D2,img_size=(64,64))

        return D2,H2

    # Stage 3: Estimate homography transformation between the reference image I3R and the warped target image bI3T.
    def stage3(self,image3R, image3T, model,H2,D2, scale=1.0):

        warped_image3T =self.warp_image(image3T, torch.inverse(H2*scale))
        displacement_output  = model(x2=warped_image3T,x1=image3R,stage=3)

        #displacement_output = displacement_output * MAIN_RHO // scale
        displacement_output = displacement_output.reshape((-1,4, 2))

        D3 = D2*2 + displacement_output
        H3 = estimate_homography(D3,img_size=(128,128))

        return D3,H3

    # Main function for the entire MS2CA-HENet architecture
    def forward(self,image3R,image3T):
        image3R = image3R.float()
        image3T = image3T.float()
        image3R.requires_grad = True
        image3T.requires_grad = True

        image2R,image2T,image1R,image1T = self.prep_images(image3R,image3T)
        #print(image1R)

        # Stage 1: Estimate displacement D1 and homography matrix H1
        D1, H1 = self.stage1(image1R, image1T, self.model)

        # Stage 2: Estimate homography transformation H2
        D2,H2 = self.stage2(image2R, image2T, self.model,H1, D1,scale=2)

        # Stage 3: Estimate homography transformation H3
        D3,H3 = self.stage3(image3R, image3T,self.model, H2, D2,scale=2) #think about scale
        
        return H1, H2, H3,D3,image2R,image2T,image1R,image1T




if __name__ == "__main__":
    net = MyNetwork()
    IR = torch.randn(1, 1, 128, 128)
    IT = torch.randn(1, 1, 128, 128)

    
    output = net(IR, IT)

    #print(output)

