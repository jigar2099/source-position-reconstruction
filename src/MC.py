import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.mcp_img import mcp_img as ngi
from tqdm import tqdm
import random


def poisson_noise_grid(pixel, A_s, A_b):
    label_g1 = np.zeros((pixel, pixel))
    noise = A_b*np.random.poisson(10,size=(pixel,pixel))
    return noise, label_g1

def perlin_noise_grid(pixel,A_s, A_b, cov=1.39):
    '''Perlin noise cosidered as 10 times larger
    in magnitude than poisson noise, however
    can be changed based on assumptions'''
    label_g1 = np.zeros((pixel, pixel))
    mu1 = mu2 = np.random.randint(0, pixel - 1, 1)[0]
    cov_mat00 = cov_mat11 = cov
    q = ngi(pixel, mu1, mu2, cov_mat00, cov_mat11, A_s, A_b)
    perlin = 10*A_b*(q.Perlin_noise())
    return perlin, label_g1

def apply_noise_selector(src, pixel, A_s, A_b, perlin=False, poisson=False):
    if perlin is True and poisson is True:
        perlin_noise, per_l = perlin_noise_grid(pixel,A_s, A_b)
        poisson_noise, poi_l = poisson_noise_grid(pixel,A_s, A_b)
        src += perlin_noise + poisson_noise
    elif perlin is True and poisson is False:
        perlin_noise, per_l = perlin_noise_grid(pixel, A_s, A_b)
        src += perlin_noise
    elif perlin is False and poisson is True:
        poisson_noise, poi_l = poisson_noise_grid(pixel, A_s, A_b)
        #print('-----------poisson loaded----------')
        src +=  poisson_noise
    elif perlin is False and poisson is False:
        perlin_noise, per_l = perlin_noise_grid(pixel, A_s, A_b)
        poisson_noise, poi_l = poisson_noise_grid(pixel, A_s, A_b)
        src += 0*perlin_noise + 0*poisson_noise
    return src

def single_source_grid(pixel, A_s, A_b, cov=1.39, edge=1, perlin=False, poisson=False, individual_bg=False):
    mu1 = mu2 = np.random.randint(0, pixel - edge, 1)[0]
    cov_mat00 = cov_mat11 = cov
    q = ngi(pixel, mu1, mu2, cov_mat00, cov_mat11, A_s, A_b)
    pos, mu, cov_mat, bg_grid = q.base()
    src, label_grid = q.noisy_gaussian_multivariate(pos, mu, cov_mat, A_s, A_b)
    if individual_bg is True:
        src = apply_noise_selector(src,pixel,A_s,A_b,perlin=perlin,poisson=poisson)

    return src, label_grid

def multi_source_grid(s_num, src, label, pixel, A_s, A_b, cov=1.2, edge=1, pow_idx=1.3, randomness=True,
                      perlin=False, poisson=False, individual_bg=False):
    mu1 = mu2 = np.random.randint(0, pixel - edge, 1)[0]
    cov_mat00 = cov_mat11 = cov
    I = np.random.randint(0, pixel, s_num - 1)#mu1
    J = np.random.randint(0, pixel, s_num - 1)#mu2

    if randomness is True:
        amp_cand = 1 * np.arange(1, 5)
        amp0 = np.array([1 / pow(L, pow_idx) for L in amp_cand])
        d = np.random.dirichlet(np.ones(amp0.shape[0]), size=1)
        amp = np.random.choice(amp0, s_num, p=d.flatten())
    elif randomness is False:
        amp_cand = 1 * np.arange(1, s_num)
        amp = np.array([1 / pow(L, pow_idx) for L in amp_cand])
    if individual_bg is True:
        for i,j,k in zip(I,J,amp):
            MU = np.array([i, j])
            COV = cov_mat00
            q = ngi(pixel, i, j, cov_mat00, cov_mat00, k, A_b)
            pos, mu, cov_mat, bg_grid = q.base()
            src, label = q.manual_addition(pos, MU, COV, k, src, label)
            src = apply_noise_selector(src,pixel,A_s,k*A_b,perlin=perlin,poisson=poisson)
    elif individual_bg is False:
        for i,j,k in zip(I,J,amp):
            MU = np.array([i, j])
            COV = cov_mat00
            q = ngi(pixel, i, j, cov_mat00, cov_mat00, k, A_b)
            pos, mu, cov_mat, bg_grid = q.base()
            src, label = q.manual_addition(pos, MU, COV, k, src, label)
        src = apply_noise_selector(src,pixel,A_s,A_b,perlin=perlin,poisson=poisson)

    return src, label


def plot_src_label(s,l):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(s)

    axarr[0].set_title(f"{i}")
    axarr[1].imshow(l)
    divider = make_axes_locatable(axarr[0])
    cax = divider.append_axes("right", size="5%", pad=0.03)
    f.colorbar(axarr[0].imshow(s), cax=cax)
    divider1 = make_axes_locatable(axarr[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.03)
    f.colorbar(axarr[1].imshow(l), cax=cax1)
    f.tight_layout()
    plt.show()


# if __name__ == '__main__':
#     num_src = 30
#     examples = 500
#     pixel = 32
#     A_s = 1
#     A_b = 0.0070#0.0010
#     per = True
#     poi = False
#     indiv_noise = False

#     octave = str(6) + "oct"
#     oct_seq = "1_512_128_064_032_016"
#     typ = "train"
#     src_file_name = f"{typ}_{octave}_{oct_seq}_SRC_{num_src}_exe_{examples}.npy"
#     label_file_name = f"{typ}_{octave}_{oct_seq}_LABEL_{num_src}_exe_{examples}.npy"
#     src_save_path = './src/noise_7e-3/'
#     label_save_path = './labels/noise_7e-3/'

#     s_lst = []
#     l_lst = []
#     for i in tqdm(range(num_src)):
#         for j in range(examples):
#             if i == 0:
#                 s, l = perlin_noise_grid(pixel, A_s, A_b)
#                 #plot_src_label(s, l)
#                 s_lst.append(s)
#                 l_lst.append(l)
#             elif i > 0:

#                 slg_s, slg_l = single_source_grid(pixel, A_s, A_b, cov=1.39, edge=1,
#                                                   perlin=per, poisson=poi, individual_bg=indiv_noise)
#                 mul_s, mul_l = multi_source_grid(i, slg_s, slg_l, pixel, A_s, A_b, cov=1.39, edge=1, pow_idx=1.3,
#                                                  perlin=per, poisson=poi, individual_bg=indiv_noise, randomness=True)
#                 #plot_src_label(mul_s, mul_l)
#                 s_lst.append(mul_s)
#                 l_lst.append(mul_l)
#     SOURCE = np.array(s_lst)
#     LABEL = np.array(l_lst)


#     #print(src_save_path+src_file_name)

#     if indiv_noise is True:
#         if per is True and poi is True:
#             src_file_name = 'indiv_per_poi_' + src_file_name
#             label_file_name = 'indiv_per_poi_' + label_file_name
#         elif per is False and poi is True:
#             src_file_name = 'indiv_poi_' + src_file_name
#             label_file_name = 'indiv_poi_' + label_file_name
#         elif per is True and poi is False:
#             src_file_name = 'indiv_per_' + src_file_name
#             label_file_name = 'indiv_per_' + label_file_name
#         elif per is False and poi is False:
#             src_file_name = 'indiv_noiseless_' + src_file_name
#             label_file_name = 'indiv_noiseless_' + label_file_name
#     elif indiv_noise is False:
#         if per is True and poi is True:
#             src_file_name = 'no_indiv_per_poi_' + src_file_name
#             label_file_name = 'no_indiv_per_poi_' + label_file_name
#         elif per is False and poi is True:
#             src_file_name = 'no_indiv_poi_' + src_file_name
#             label_file_name = 'no_indiv_poi_' + label_file_name
#         elif per is True and poi is False:
#             src_file_name = 'no_indiv_per_' + src_file_name
#             label_file_name = 'no_indiv_per_' + label_file_name
#         elif per is False and poi is False:
#             src_file_name = 'no_indiv_noiseless_' + src_file_name
#             label_file_name = 'no_indiv_noiseless_' + label_file_name

#     print(src_save_path + src_file_name)
#     np.save(src_save_path+src_file_name, SOURCE)
#     np.save(label_save_path+label_file_name, LABEL)

