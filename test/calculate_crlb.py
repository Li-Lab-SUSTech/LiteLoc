from utils.help_utils import calculate_crlb_rmse
import torch

loc_model = torch.load("../training_model/liteloc_vector_simu_astig_ph4kt6k_0409/checkpoint.pkl")

calculate_crlb_rmse(loc_model)