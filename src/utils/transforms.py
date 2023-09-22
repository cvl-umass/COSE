import skimage
import numpy as np
import skimage
from lime import lime_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import saliency.core as saliency

from skimage.metrics import structural_similarity as ssim
from skimage.morphology import binary_erosion
from scipy.stats import spearmanr
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from loguru import logger

def get_reverse_affine_explanations(rotated_explanation, rotated_ones_mask, rot_val, scale_val, trans_val):
    # Reverse affine transform
    r_tform = skimage.transform.AffineTransform(
        translation=-1*trans_val,
        rotation=-1*rot_val,
        scale=1/scale_val,
    )
    r_t_explanation = skimage.transform.warp(
        rotated_explanation,
        r_tform.inverse,
        mode="constant",
        cval=0,
        preserve_range=True,
    )
    r_t_ones_mask = skimage.transform.warp(
        rotated_ones_mask,
        r_tform.inverse,
        mode="constant",
        cval=0,
        preserve_range=True,
    )
    return r_t_explanation, r_t_ones_mask

def get_similarity(explanation, t_explanation, method="ssim", mask=None):
    if method == "ssim":
        # Window size to 5 following "Sanity Checks for Saliency Maps"
        drange = max(t_explanation.max() - t_explanation.min(), explanation.max() - explanation.min())
        ssim_val, ssim_full = ssim(explanation, t_explanation, win_size=5, full=True, data_range=drange)
        if mask is not None:
            # Erode the mask to only use areas which have no invalid pixels
            mask = binary_erosion(mask, footprint=np.ones((5, 5)))
            return np.mean(ssim_full[mask])
        return ssim_val
    elif method == "spearman":
        if mask is not None:
            return spearmanr(explanation[mask], t_explanation[mask], axis=None).correlation
        return spearmanr(explanation, t_explanation, axis=None).correlation
    else:
        raise NotImplementedError(f"Method specified [{method}] not implemented")

def get_reverse_rot_explanation(rotated_explanation, rotated_ones, rot_val):
    unrot_explanation = skimage.transform.rotate(rotated_explanation, -rot_val, preserve_range=True, resize=False, clip=False, mode='constant', cval=0)
    unrot_ones = skimage.transform.rotate(rotated_ones, -rot_val, preserve_range=True, resize=False, clip=False, mode='constant', cval=0)
    return unrot_explanation, unrot_ones

def explain_img(i, inputs, t_type, batch_predict, idx2label, target_class, config, model_name, **kwargs):
    
    np_img = inputs.cpu().detach().numpy()

    t_test_pred = batch_predict(np_img[None])
    t_pred_idx = t_test_pred.squeeze().argmax()
    t_pred_class = idx2label[t_pred_idx]

    np_img = np.moveaxis(np_img, 0, 2)

    if config["method"] == "lime":
        transformed_explainer = lime_image.LimeImageExplainer()
        transformed_explanation = transformed_explainer.explain_instance(
            np_img,
            batch_predict, # classification function
            top_labels=5, 
            hide_color=0, 
            random_seed=config['random_seed'],
            num_samples=1000,
        ) # number of images that will be sent to classification function

        # Shade areas that contribute to top prediction
        transformed_temp2, transformed_mask2 = transformed_explanation.get_image_and_mask(transformed_explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        transformed_temp2 = (transformed_temp2-np.min(transformed_temp2))/(np.max(transformed_temp2)-np.min(transformed_temp2))
        explanation_img = mark_boundaries(transformed_temp2, transformed_mask2)
        explanation = transformed_mask2.astype(float)

    elif config["method"] in ["gradcam", "gradcamPP"]:
        cam = kwargs["cam"]
        grayscale_cam = cam(input_tensor=inputs.unsqueeze(0), targets=[ClassifierOutputTarget(t_pred_idx)])
        grayscale_cam = grayscale_cam[0, :]
        explanation_img = show_cam_on_image((np_img-np.min(np_img))/(np.max(np_img)-np.min(np_img)), grayscale_cam, use_rgb=True)
        explanation = grayscale_cam
    
    elif config["method"] in ["ig", "smoothgrad", "guided_ig", "blur_ig"]:
        call_model_args = {"class_idx_str": t_pred_idx}
        call_model_function = kwargs["call_fn"]
        baseline = np.zeros(np_img.shape)
        if config["method"] == "guided_ig":
            guided_ig = saliency.GuidedIG()
            # Default from saliency repo, cited in guided IG paper
            explanation = guided_ig.GetMask(np_img, call_model_function, call_model_args, x_steps=200, x_baseline=baseline, max_dist=0.02, fraction=0.25)
        elif config["method"] == "blur_ig":
            blur_ig = saliency.BlurIG()
            # Official repo has 50 steps, we are using 100 from saliency repo default
            explanation = blur_ig.GetMask(np_img, call_model_function, call_model_args, batch_size=20)
        elif config["method"] == "ig":
            integrated_gradients = saliency.IntegratedGradients()
            # Paper says 20-300, we use saliency repo default
            explanation = integrated_gradients.GetMask(np_img, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
        elif config["method"] == "smoothgrad":
            # Using vanilla gradients
            gradient_saliency = saliency.GradientSaliency()
            # 10-20% recommended in paper, we use 15%. nsamples >= 50 from paper
            explanation = gradient_saliency.GetSmoothedMask(np_img, call_model_function, call_model_args, stdev_spread=.15, nsamples=50)
        # Saliency map is the visualization
        explanation = saliency.VisualizeImageGrayscale(explanation)
        explanation_img = show_cam_on_image((np_img-np.min(np_img))/(np.max(np_img)-np.min(np_img)), explanation, use_rgb=True)
    
    if config['to_save_imgs']:
        plt.imshow(explanation_img)
        plt.savefig(f"../image_outputs/{config['dataset']}/{model_name}/{i:02d}_t{target_class}_p{t_pred_class}_{t_type}_{kwargs.get('mag', 0):.2f}_{config['method']}.jpg")
        plt.close()

    return t_pred_class, explanation
