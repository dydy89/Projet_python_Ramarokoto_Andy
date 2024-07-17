import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TableDetector:
    def __init__(self, model_name='facebook/detr-resnet-50'):
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.processor = DetrImageProcessor.from_pretrained(model_name)
    
    def predict(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits, outputs.pred_boxes

    def extract_table(self, image_path, threshold=0.1):  # Réduire encore plus le seuil
        logits, bboxes = self.predict(image_path)
        probas = logits.softmax(-1)[..., :-1].max(-1)
        keep = probas.values > threshold
        print(f"Detection probabilities: {probas.values}")  # Afficher les probabilités de détection
        bboxes = bboxes[keep]
        print(f"Bounding boxes: {bboxes}")  # Afficher les boîtes englobantes
        return bboxes, keep

    def visualize_predictions(self, image_path, bboxes, keep):
        image = Image.open(image_path)
        width, height = image.size
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for bbox in bboxes:
            # Convertir les coordonnées normalisées en coordonnées de l'image
            xmin, ymin, xmax, ymax = bbox.detach().numpy()  # Détacher et convertir en NumPy
            rect = patches.Rectangle((xmin * width, ymin * height), (xmax - xmin) * width, (ymax - ymin) * height,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

if __name__ == "__main__":
    # Exemple d'utilisation de TableDetector
    sample_image = r'C:\Users\33768\Pictures\une_facture.jpg'
    detector = TableDetector()
    bboxes, keep = detector.extract_table(sample_image, threshold=0.1)  # Utiliser un seuil encore plus réduit
    detector.visualize_predictions(sample_image, bboxes, keep)  # Visualiser les prédictions
    print(f"Tables detected: {bboxes}")
