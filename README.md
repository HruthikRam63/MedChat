# MedChat: Your Comprehensive Medical Information Assistant
<h4>MedChat is your go-to solution for accessing and managing medical information seamlessly. From adverse drug reactions to pharmacy locations, MedChat offers reliable and instant assistance!</h4>

## Project Description
<br clear="both">

<img align="right" height="250" src="https://i.postimg.cc/Fs4dFcqD/Med-Chat-icon.jpg" alt="MedChat Logo" />

**MedChat** is a sophisticated chatbot designed to provide users with essential medical information. It answers queries related to adverse drug reactions, monitors blood pressure, provides hospital data, and locates pharmacies. By leveraging advanced natural language processing (NLP) and machine learning models, MedChat ensures accurate and helpful responses to a wide range of medical inquiries. This project combines user-friendly design with powerful backend algorithms to deliver an engaging and informative experience.



## NLP Integration

**MedChat** employs Natural Language Processing (NLP) to understand and process user queries effectively. The following NLP techniques and components are utilized:

1. **Tokenization:** 
   - **Description:** Converts user input into individual tokens (words or phrases) for analysis.
   - **Implementation:** NLTK’s `word_tokenize` function is used to split the input text into manageable tokens.

2. **Lemmatization:**
   - **Description:** Reduces words to their base or root form to ensure uniformity in understanding.
   - **Implementation:** NLTK’s `WordNetLemmatizer` is applied to standardize words by removing suffixes.

3. **Bag of Words Model:**
   - **Description:** Transforms the tokenized input into a numerical format that the machine learning model can process.
   - **Implementation:** A binary representation is created where the presence of each word from the vocabulary is marked with a `1` or `0`.

4. **Intent Classification:**
   - **Description:** Determines the user’s intent based on the processed input.
   - **Implementation:** The processed input is fed into a neural network model, which classifies the input into predefined categories.

5. **Model Training and Prediction:**
   - **Description:** Uses historical data to train the model to understand and predict user queries.
   - **Implementation:** A neural network is trained on labeled data to predict user intents and provide accurate responses.

The integration of these NLP techniques ensures that MedChat can accurately interpret and respond to a diverse range of medical queries, enhancing the overall user experience.


## Table of Contents

1. [Project Description](#project-description)
2. [NLP Integration](#nlp-integration)
3. [Features](#features)
4. [Requirements](#requirements)
5. [Installation Instructions](#installation-instructions)
6. [Usage Details](#usage-details)
7. [Challenges Faced](#challenges-faced)
8. [Contributions](#contributions)
9. [Further Development](#further-development)
10. [Contact Information](#contact-information)
11. [Screenshots](#screenshots)

## Features

- **Adverse Drug Reactions:** Get information on possible side effects and interactions of various medications.
- **Blood Pressure Tracking:** Monitor and analyze blood pressure readings.
- **Hospital Data:** Access information about nearby hospitals and their services.
- **Pharmacy Locations:** Find the nearest pharmacies based on your location.
- **User-Friendly Interface:** Easy-to-navigate chat interface for a seamless user experience.

## Requirements

- **Programming Languages:** Python
- **Libraries/Frameworks:** TensorFlow/Keras, NLTK, Tkinter
- **Data Files:**
  - `data/prompts_n_responses.json`: Dataset containing prompts and responses.
  - `data/tokens.pkl`: Pickled file with tokenized words.
  - `data/labels.pkl`: Pickled file with class labels.
- **Model File:** `models/model.h5`

## Installation Instructions

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/MedChat.git
    ```
2. **Navigate to the Project Directory:**
    ```bash
    cd MedChat
    ```
3. **Install Dependencies:**
    Ensure you have Python installed. Create a virtual environment and install the required libraries.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
4. **Run the Training Script:**
    To train the model, execute:
    ```bash
    python src/training.py
    ```
5. **Launch the GUI Application:**
    ```bash
    python src/medchat_gui.py
    ```

## Usage Details

1. **Initial Interaction:**
   Upon launching MedChat, users are greeted with a chat interface where they can input their medical-related queries.

2. **Query Handling:**
   Users can ask questions about:
   - Adverse drug reactions.
   - Blood pressure monitoring.
   - Hospital information.
   - Pharmacy locations.

3. **Receiving Responses:**
   MedChat processes the queries and returns relevant information based on the predefined data and trained model.

4. **Interface Elements:**
   - **Chat Log:** Displays ongoing conversation.
   - **Entry Box:** Area for users to type their queries.
   - **Send Button:** Sends the user's message to MedChat.

## Challenges Faced

1. **Data Handling:**
   - **Challenge:** Ensuring the accuracy and comprehensiveness of medical information.
   - **Solution:** Used a well-curated dataset and continuous updates to the model to maintain accuracy.

2. **Model Training:**
   - **Challenge:** Balancing the model's ability to generalize across various medical queries.
   - **Solution:** Employed techniques such as data augmentation and hyperparameter tuning to improve model performance.

3. **User Interaction:**
   - **Challenge:** Designing an intuitive interface that effectively communicates medical information.
   - **Solution:** Implemented user feedback mechanisms to refine the interface and improve user experience.

4. **Integration of Medical Data:**
   - **Challenge:** Integrating diverse medical datasets and ensuring their relevance.
   - **Solution:** Regularly updated datasets and collaborated with medical professionals to validate information.

## Contributions

Contributions to MedChat are highly encouraged! To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Commit your changes and push to your branch.
4. Open a pull request with a detailed description of your changes.

## Further Development

Future enhancements for MedChat include:
- **Expanded Knowledge Base:** Adding more medical topics and detailed responses.
- **Enhanced Model Accuracy:** Implementing advanced NLP techniques for better understanding of user queries.
- **User Personalization:** Incorporating user preferences and historical data to tailor responses more effectively.
- **Mobile Compatibility:** Developing a mobile version of the application for broader accessibility.

## Contact Information

For any questions or inquiries, please contact:
- **Email:** hruthikram63@gmail.com
- **LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/in/hruthikram63)

## Screenshots

<p align="center">
    <img src="https://i.postimg.cc/gX58ZgtY/Screenshot-2024-08-10-105322.png" alt="Screenshot 1" width="250"/>
    <img src="https://i.postimg.cc/30vpC2z2/Screenshot-2024-08-10-105413.png" alt="Screenshot 2" width="250"/>
    <img src="https://i.postimg.cc/w3yhpd33/Screenshot-2024-08-10-105452.png" alt="Screenshot 3" width="250"/>
    <img src="https://i.postimg.cc/zVTKySCr/Screenshot-2024-08-10-105515.png" alt="Screenshot 4" width="250"/>
    <img src="https://i.postimg.cc/14kDRYNB/Screenshot-2024-08-10-105539.png" alt="Screenshot 5" width="250"/>
    <img src="https://i.postimg.cc/4YZtV7Ph/Screenshot-2024-08-10-105600.png" alt="Screenshot 6" width="250"/>
</p>
