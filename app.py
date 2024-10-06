import streamlit as st
from chat import OpenAIModelText
from chat import OpenAIModelImage

# Initialize the OpenAI text model
openai_model_text = OpenAIModelText()
openai_model_text.load_model()  # Load the specified model for text

# Initialize the OpenAI image model
openai_model_image = OpenAIModelImage()
openai_model_image.load_model(model_id="dall-e-2")  # Load the specified model for image generation

# Set the title
st.title("Simple chat")

# Add custom CSS for autumn colors
st.markdown(
    """
    <style>
    .chat-message {
        background-color: #F9E2B5; /* Light yellow background */
        color: #3E2723; /* Dark brown text */
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .user {
        background-color: #D68B3B; /* Orange background for user */
        color: white;
    }
    .assistant {
        background-color: #A65E2B; /* Dark orange background for assistant */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a sidebar menu
menu_options = ["Chat", "Generate Image"]
selected_option = st.sidebar.selectbox("Select an option", menu_options)

if selected_option == "Chat":
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], key=message["content"]):  # Use key to avoid warnings
            st.markdown(message["content"], unsafe_allow_html=True)

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=True)

        # Generate response using OpenAI model
        system_message = "You are a helpful assistant."  # Define system behavior
        response = openai_model_text.infer(system_message, prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

elif selected_option == "Generate Image":
    # Create a text input for image generation
    image_prompt = st.text_input("Enter a description for the image:")
    
    if st.button("Generate Image"):
        if image_prompt:
            # Generate image using OpenAI model
            image_url = openai_model_image.infer(input_data=image_prompt)
            # Display the generated image
            st.image(image_url, caption=image_prompt, use_column_width=True)
        else:
            st.warning("Please enter a description for the image.")