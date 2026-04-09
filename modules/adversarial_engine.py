import tensorflow as tf
import numpy as np

class AdversarialEngine:
    def __init__(self, model):
        """
        Adversarial Resilience Engine for A-DIDS.
        
        Args:
            model (tf.keras.Model): The IDS model to attack/harden.
        """
        self.model = model
        self.loss_object = tf.keras.losses.BinaryCrossentropy()

    def create_adversarial_pattern(self, input_image, input_label):
        """
        Calculates the gradients for FGSM.
        """
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = self.model(input_image)
            loss = self.loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation.
        signed_grad = tf.sign(gradient)
        return signed_grad

    def generate_fgsm_sample(self, input_data, epsilon=0.01):
        """
        Generates an adversarial example using FGSM.
        
        Args:
            input_data (np.ndarray): Original network flow data.
            epsilon (float): Perturbation step size.
            
        Returns:
            np.ndarray: The perturbed adversarial sample.
        """
        # Ensure input is a tensor
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        
        # Get model prediction to determine target
        # For evasion, we want to shift 'Attack' (1) towards 'Normal' (0)
        # or just maximize the error.
        prediction = self.model(input_tensor)
        
        # If the sample is an attack, we want to fool it into being predicted as benign.
        # So we use the "Attack" label as the ground truth and add the sign of gradient 
        # to maximize the loss (push it away from 1 towards 0).
        label = tf.ones_like(prediction) 
        
        perturbations = self.create_adversarial_pattern(input_tensor, label)
        adversarial_sample = input_tensor + epsilon * perturbations
        
        # Clipping is less common in tabular data than images, but good for stability
        # adversarial_sample = tf.clip_by_value(adversarial_sample, -3, 3) 
        
        return adversarial_sample.numpy()

    def adversarial_training_step(self, x_batch, y_batch, optimizer, epsilon=0.01):
        """
        Performs a single adversarial training step.
        """
        with tf.GradientTape() as tape:
            # Generate adversarial examples for this batch
            x_adv = self.generate_fgsm_sample(x_batch, epsilon=epsilon)
            
            # Predict both original and adversarial
            p_orig = self.model(x_batch)
            p_adv = self.model(x_adv)
            
            # Combined loss: optimize for both accuracy and robustness
            loss_orig = self.loss_object(y_batch, p_orig)
            loss_adv = self.loss_object(y_batch, p_adv)
            total_loss = (loss_orig + loss_adv) / 2
            
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return total_loss

if __name__ == "__main__":
    print("A-DIDS Adversarial Engine initialized. Ready for Red-Teaming.")
