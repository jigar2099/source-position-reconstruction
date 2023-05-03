import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

class mcp_img:
    
    def __init__(self, pixel, mu1, mu2, cov_mat00, cov_mat11, src, bg):
        self.pixel = pixel
        self.mu1 = mu1
        self.mu2 = mu2
        self.cov_mat00 = cov_mat00
        self.cov_mat11 = cov_mat11
        self.src = src
        self.bg = bg
        
    def base(self):
    # introduce grid-base for image
        x = np.linspace(0,self.pixel-1,self.pixel)
        y = np.linspace(0,self.pixel-1,self.pixel)
        x, y = np.meshgrid(x, y)#-------------->establishment of grids for image
        pos = np.dstack((x,y))#----------------> dstacking provides stacking of x and y grids on top of each other

        # generate matrix of mean and covariance_matrix
        mu = np.array([self.mu1,self.mu2])
        cov_mat = np.array([[self.cov_mat00, 0],[0, self.cov_mat11]])
        bg_grid = np.dstack((x,y))
    
        return pos, mu, cov_mat, bg_grid

    def noisy_gaussian_multivariate(self, pos, mu, cov_mat,src,bg):
        n = mu.shape[0]
        det_sig = np.linalg.det(cov_mat)
        inv_sig = np.linalg.inv(cov_mat)

        powerr = np.einsum('ijk,kl,ijl->ij', pos-mu, inv_sig, pos-mu)
        denominator = np.sqrt(np.power((2*np.pi),n) * det_sig)
        out = 1*src * np.exp(-powerr / 2) / denominator
        bg_grid = 0*int(src/16)*np.random.poisson(bg,size=out.shape)
        label_grid = np.zeros((out.shape))
        label_grid[mu[0],mu[1]] +=1
        return out, label_grid

    def manual_addition(self, pos, mu, cov_val, A_src, img, label_grid):
        cov_mat = np.array([[cov_val, 0],[0, cov_val]])
        n = mu.shape[0]
        det_sig = np.linalg.det(cov_mat)
        inv_sig = np.linalg.inv(cov_mat)

        powerr = np.einsum('ijk,kl,ijl->ij', pos-mu, inv_sig, pos-mu)
        #print('powerr:',powerr.shape)
        #powerr = np.einsum('...k,kl,...l->...', pos-mu, inv_sig, pos-mu)fig.update_layout(title_text="Ring cyclide")
        denominator = np.sqrt(np.power((2*np.pi),n) * det_sig)
        out = 1*A_src*(np.exp(-powerr/2)/denominator) #+ img
        label_grid[mu[1],mu[0]] +=1
        output = out+img
        #print(A_src,out.shape,np.max(out))
        #return out, label_grid
        return output, label_grid
    
    def Perlin_noise(self):

        noise1 = PerlinNoise(octaves=4)
        noise2 = PerlinNoise(octaves=8)
        noise3 = PerlinNoise(octaves=16)
        noise4 = PerlinNoise(octaves=32)
        noise5 = PerlinNoise(octaves=64)
        noise6 = PerlinNoise(octaves=128)

        xpix, ypix = self.pixel, self.pixel
        pic = []
        for i in range(xpix):
            row = []
            for j in range(ypix):
                noise_val = noise1([i/xpix, j/ypix])
                #noise_val += 0.5 * noise2([i/xpix, j/ypix])
                #noise_val += 0.128 * noise3([i/xpix, j/ypix])
                #noise_val += 0.128 * noise4([i/xpix, j/ypix])
                #noise_val += 0.128 * noise5([i/xpix, j/ypix])
                #noise_val += 0.128 * noise6([i/xpix, j/ypix])

                noise_val += 0.512 * noise2([i / xpix, j / ypix])
                noise_val += 0.128 * noise3([i / xpix, j / ypix])
                noise_val += 0.064 * noise4([i / xpix, j / ypix])
                noise_val += 0.032 * noise5([i / xpix, j / ypix])
                noise_val += 0.016 * noise6([i / xpix, j / ypix])

                row.append(noise_val)
            pic.append(row)
        pic = pic+np.abs(np.min(pic))
        #print('max of pic:{}, min of pic:{}'.format(np.max(pic),np.min(pic)))
        #plt.imshow(pic, cmap='viridis')
        #plt.colorbar()
        #plt.show()
        return pic