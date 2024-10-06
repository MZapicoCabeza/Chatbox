import openai
import os
from dotenv import load_dotenv

import openai
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables for OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Clase para manejar el modelo de OpenAI
class OpenAIModelText():
    def __init__(self):
        self.client = OpenAI()  # Initialize OpenAI client

    def load_model(self, model_name: str = 'gpt-4o-mini-2024-07-18'):
        """Cargar el modelo de OpenAI"""
        self.model_identifier = model_name
        print(f"Modelo {model_name} cargado.")

    def infer(self, system: str, input_data: str) -> str:
        """Generar respuesta basada en la entrada del usuario"""
        completion = self.client.chat.completions.create(
            model=self.model_identifier,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": input_data}
            ]
        )
        if completion.choices:
            generated_text = completion.choices[0].message.content
            return generated_text
        return "No se pudo generar respuesta."



class OpenAIModelImage():
    """
    This class handles the functionality for generating images using a specified OpenAI model.
    It inherits from the AbstractModelHandler class and provides methods to load the model
    and perform inference to generate images based on a textual prompt.
    """

    def __init__(self):
        """
        Constructor method for initializing the OpenAIModelImage class.
        This method loads environment variables from a `.env` file, typically used to store
        API keys and other sensitive information.
        """
        load_dotenv()

    def load_model(self, model_id: str):
        """
        This method initializes the model by setting the model name based on user input and
        creating a client instance of the OpenAI API. The model name is stored as a class
        attribute for use in other methods.

        Args:
            model_id (str): The name of the model to be used for image generation (e.g., "dall-e-2").
        """
        self.model_id = model_id
        self.client = OpenAI()

    def infer(self, input_data: str, image_size: str = "1024x1024", image_quality: str = "standard", number: int = 1) -> str:
        """
        This method performs image generation using the specified OpenAI model. It takes a textual
        prompt and generates an image based on that prompt.

        Args:
            input_text (str): The textual prompt describing the desired image.
            image_size (str): The size of the output image. Default is "1024x1024".
            image_quality (str): The quality of the output image. Default is "standard".
            number (int): The number of images to generate. Default is 1.

        Returns:
            str: The URL of the generated image.
        """
        completion = self.client.images.generate(
            model=self.model_id,
            prompt=input_data,
            size=image_size,
            quality=image_quality,
            n=number,
        )

        return completion.data[0].url

