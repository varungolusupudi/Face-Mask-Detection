import gradio as gr
from fastai.vision.all import *

# load exported fastai learner
learn = load_learner("models/mask_classifier.pkl")

def predict(img):
    # img comes in as a PIL Image from Gradio
    pred_class, pred_idx, probs = learn.predict(img)
    # return a dict: {label: probability}
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a face image"),
    outputs=gr.Label(num_top_classes=3, label="Class probabilities"),
    title="Face Mask Classifier",
    description="Classes: with_mask, without_mask, mask_weared_incorrect"
)

if __name__ == "__main__":
    demo.launch()
