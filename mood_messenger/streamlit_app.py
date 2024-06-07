import pandas as pd
from tqdm import tqdm
import streamlit as st
import re
from datetime import datetime
from transformers import pipeline
import time
import streamlit_shadcn_ui as ui
import matplotlib.pyplot as plt
import seaborn as sns

st.image("images/mmmm.png")
with st.expander("How to Export Whatsapp chats?"):
    st.video("images/video.mov")

st.sidebar.button("Reset", type="primary")


def read_whatsapp_chat(uploaded_file):
    chat_list = []
    for line in uploaded_file:
        chat_list.append(line.decode("utf-8").strip())
    return chat_list


def parse_chat(chat):
    pattern = r"^\[(?P<date>\d{2}/\d{2}/\d{4}), (?P<time>\d{2}:\d{2}:\d{2})\] (?P<from>[^:]+): (?P<message>.+)$"
    match = re.match(pattern, chat)
    if match:
        date_str = match.group("date")
        time_str = match.group("time")
        datetime_str = f"{date_str} {time_str}"
        datetime_obj = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
        return datetime_obj, match.group("from"), match.group("message")
    else:
        return None, None, None


# Initialize session state for storing DataFrame
if "final_dataframe" not in st.session_state:
    st.session_state.final_dataframe = None


@st.cache_resource(experimental_allow_widgets=True)
def load_model1():
    sentiment_task = pipeline(
        "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    ui.badges(
        badge_list=[("Loaded Sentiment Analysis Model from Huggingface", "default")],
        class_name="flex gap-2",
        key="badges1",
    )
    return sentiment_task


@st.cache_resource(experimental_allow_widgets=True)
def load_model2():
    classifier = pipeline(
        task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=2
    )
    ui.badges(
        badge_list=[("Loaded Emotions Model from Huggingface", "default")],
        class_name="flex gap-2",
        key="badges2",
    )
    return classifier


def sentiment_analysis(data, column, sentiment_task):
    # Initialize lists to store labels and scores
    sentiment = []
    sentiment_score = []

    # Initialize the Streamlit progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Total number of rows
    total_rows = data.shape[0]

    # Iterate over DataFrame rows and classify text
    for index, row in tqdm(
        data.iterrows(), total=total_rows, desc="Analyzing Sentiment"
    ):
        sentence = row[column]
        sentence = str(sentence)[
            :513
        ]  # Ensure the text is a string and truncate if necessary

        if pd.isna(sentence) or sentence == "":
            sentiment.append("neutral")
            sentiment_score.append(0)
        else:
            model_output = sentiment_task(sentence)
            sentiment.append(model_output[0]["label"])
            sentiment_score.append(model_output[0]["score"])

        # Update the progress bar
        progress = min((index + 1) / total_rows, 1.0)
        progress_bar.progress(progress)
        progress_text.text(f"Sentiment Analysis row {index + 1} of {total_rows}")

    progress_text.text("Sentiment analysis completed!")

    # Add labels and scores as new columns
    data[f"sentiment_{column}"] = sentiment
    data[f"sentiment_score_{column}"] = sentiment_score

    return data


def emotion_classification(data, column, classifier):
    emotion = []

    # Initialize the Streamlit progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Total number of rows
    total_rows = data.shape[0]

    for index, row in tqdm(data.iterrows(), total=total_rows, desc="Analyzing Emotion"):
        sentence = row[column]
        sentence = str(sentence)
        if pd.isna(sentence) or sentence == "":
            emotion.append("neutral")  # Assuming "neutral" for empty sentences
        else:
            model_output = classifier(sentence)
            emotion.append(model_output[0])  # Handle unexpected output format

        # Update the progress bar
        progress = min((index + 1) / total_rows, 1.0)
        progress_bar.progress(progress)
        progress_text.text(f"Emotion Analysis row {index + 1} of {total_rows}")

    progress_text.text("Emotion analysis completed!")

    # Add labels as a new column
    data[f"emotion_{column}"] = emotion

    return data


# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat file", type="txt")

if uploaded_file is not None and st.session_state.final_dataframe is None:
    # Display a progress bar while reading the file
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Read the contents of the uploaded file line by line and append to chat_list
    chat_list = []
    total_lines = sum(1 for _ in uploaded_file)
    uploaded_file.seek(0)  # Reset the file pointer to the beginning

    for i, line in enumerate(uploaded_file):
        chat_list.append(line.decode("utf-8").strip())
        progress = (i + 1) / total_lines
        progress_bar.progress(progress)
        progress_text.text(f"Loading line {i + 1} of {total_lines}")
        time.sleep(0.001)  # Add a small delay for demonstration purposes

    progress_text.text("File processing completed!")

    # Create a DataFrame from the chat_list
    df = pd.DataFrame(chat_list, columns=["chat"])

    # Parse the chat data
    df[["datetime", "from", "chat"]] = df["chat"].apply(
        lambda x: pd.Series(parse_chat(x))
    )
    data = df[["datetime", "from", "chat"]]
    data.dropna(inplace=True)

    # Load models
    sentiment_task = load_model1()
    classifier = load_model2()

    # Perform sentiment analysis
    data = sentiment_analysis(data, "chat", sentiment_task)

    # Perform emotion classification
    data = emotion_classification(data, "chat", classifier)

    # Process the emotion column
    data["emotion1"] = data["emotion_chat"].apply(lambda x: x[0]["label"])
    data["emotion2"] = data["emotion_chat"].apply(lambda x: x[1]["label"])
    data.drop(columns=["emotion_chat"], inplace=True)

    # Store the final dataframe in session state
    st.session_state.final_dataframe = data

# Use the stored DataFrame for further analysis
if st.session_state.final_dataframe is not None:
    data = st.session_state.final_dataframe

    # Display the DataFrame
    # st.subheader("Chat DataFrame")
    # st.write(data)

    # Add further analysis here
    user_list = data["from"].unique()
    selected_user = st.sidebar.selectbox("Select a user", user_list)

    selected_user_data = data[data["from"] == selected_user]

    page = st.sidebar.radio(
        "Select a Page",
        [
            "User Activity",
            "Sentiment Analysis Overview",
            "Mean Daily Sentiment",
            "Emotion Analysis",
        ],
    )

    if page == "User Activity":
        st.subheader(f"User Activity - {selected_user}")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=data, y="from", color="#8e5575")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.yaxis.grid(False)
        plt.xlabel("Count")
        plt.ylabel("")
        plt.title("User Activity", loc="right")
        plt.tight_layout()
        st.pyplot(plt)

        # Display messages from the selected user
        st.write(f"Messages from {selected_user}:")
        st.write(data[data["from"] == selected_user])

    if page == "Sentiment Analysis Overview":
        pass

    if page == "Mean Daily Sentiment":
        pass

    if page == "Emotion Analysis":
        st.subheader(f"Emotion Analysis - {selected_user}")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.countplot(data=selected_user_data, y="emotion1", color="#55a058", alpha=0.7)
        sns.countplot(data=selected_user_data, y="emotion2", color="#ec8b33", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.yaxis.grid(False)
        plt.xlabel("Count")
        plt.ylabel("")
        plt.xlim(0, 200)
        plt.title("Emotion Analysis", loc="right")
        plt.tight_layout()
        st.pyplot(plt)

        # Display messages from the selected user
        st.write(f"Messages from {selected_user}:")
        st.write(data[data["from"] == selected_user])
