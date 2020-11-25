import sys
import logging

from pytorch_cnn_visualization.misc_functions import get_example_params, convert_to_grayscale, save_gradient_images
from pytorch_cnn_visualization.vanilla_backprop import VanillaBackprop
from pytorch_cnn_visualization.gradcam import GradCam
from pytorch_cnn_visualization.guided_gradcam import GuidedBackprop as GuidedBackpropCam
from pytorch_cnn_visualization.guided_gradcam import guided_grad_cam
from pytorch_cnn_visualization.smooth_grad import generate_smooth_grad


def main():
    target_example = int(sys.argv[1])  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) = \
        get_example_params(target_example)
    
    # Grad cam
    gcv2 = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = gcv2.generate_cam(prep_img, target_class)
    logging.info('Grad cam completed')
    
    # Guided backprop
    GBP = GuidedBackpropCam(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    logging.info('Guided backpropagation completed')
    
    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
    logging.info('Guided grad cam completed')
    
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    logging.info('Vanilla backprop completed')
    
    VBP = VanillaBackprop(pretrained_model)
    # GBP = GuidedBackprop(pretrained_model)  # if you want to use GBP dont forget to
    # change the parametre in generate_smooth_grad

    param_n = 50
    param_sigma_multiplier = 4
    smooth_grad = generate_smooth_grad(VBP,  # ^This parameter
                                       prep_img,
                                       target_class,
                                       param_n,
                                       param_sigma_multiplier)

    # Save colored gradients
    save_gradient_images(smooth_grad, file_name_to_export + '_SmoothGrad_color')
    # Convert to grayscale
    grayscale_smooth_grad = convert_to_grayscale(smooth_grad)
    # Save grayscale gradients
    save_gradient_images(grayscale_smooth_grad, file_name_to_export + '_SmoothGrad_gray')
    logging.info('Smooth grad completed')
    
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)

    # Make sure dimensions add up!
    grad_times_image = vanilla_grads * prep_img.detach().numpy()[0]
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads,
                         file_name_to_export + '_Vanilla_grad_times_image_gray')
    logging.info('Grad times image completed.')
    
    
if __name__ == "__main__":
    main()