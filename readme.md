# Summary

Goal of the project is to identify machine learning models for signature verification and convert them to the ONNX format

We use two models:
- Signature Image Cleaning
- Siamese signature verification

![Overview - Signature cleaning](/support-images/signature_overview.png)
![Overview - Signature verification](/support-images/signature_overview_verification.png)  


I've used two notebooks published on Kaggle.

# Prerequisites

- To analyze models graph we use netron. Here you can download it:
    > https://netron.app
- Python >= 3.10

# Signature Image Cleaning

Signature cleaning seeks to automatically remove background noise, text, lines etc that often accompany signatures in the wild. 

To clean images I've used this notebook:
> https://www.kaggle.com/code/victordibia/signature-image-cleaning-with-tensorflow-2-0

All models released by this notebook are in "signature-image-cleaning" folder. 
Files have been downloaded from here:
> https://www.kaggle.com/code/victordibia/signature-image-cleaning-with-tensorflow-2-0/data

Downloaded model is in SavedModel format by Tensorflow.

To convert this model you need to install tensorflow-onnx:
> https://github.com/onnx/tensorflow-onnx

From shell, from "signature-image-cleaning" folder type this:
> pip install tensorflow
> 
> pip install onnxruntime
> 
> pip install -U tf2onnx
> 
> python -m tf2onnx.convert --saved-model ./model --output model.onnx

Is we open converted model we have this:

![Model graph after conversion](/support-images/graph_sig_cleaning.png)

We want to:
- change input and output names
- change batch size unk_127 and unk_128. Value we know is 1

We need to execute python script:
> python convert.py

Now we have:

![Model graph after script execution](/support-images/graph_sig_cleaning_final.png)

# Siamese signature verification

Siamese and contrastive loss can be used for signature verification by training a Siamese neural network to differentiate between genuine and forged signatures. The network consists of two identical branches, one for each of the two signatures being compared. The output of the two branches is then fed into a contrastive loss function, which calculates the difference between the two signatures and penalizes the network if the difference is too small (indicating that the signatures are likely to be genuine) or too large (indicating that the signatures are likely to be forged).

We use this notebook on Kaggle:
> https://www.kaggle.com/code/medali1992/siamese-signature-verification-with-confidence

Output model can be found here:
> https://www.kaggle.com/code/medali1992/siamese-signature-verification-with-confidence/data?select=convnet_best_loss.pt

All files are in "signature-verification" folder.

Model input is a in pytorch format, to convert it in onnx we'll use TORCH.ONNX:
> https://pytorch.org/docs/stable/onnx.html

We have to install these dependencies:

> pip install torch torchvision
> 
> pip install scikit-learn
>
> pip install onnxruntime

To convert model to ONNX, run this command:
> python convert.py

We'll have this:

![Model graph after conversion](/support-images/graph_sig_verification_final.png)

# End

We can use converted models on out applications. We'll develop an .Net SDK to verify signatures.