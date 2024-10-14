import json
import os
from textwrap import dedent

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def send_message(text: str, channel: str) -> bool:
    try:
        slack_token = os.environ["SLACK_ACCESS_TOKEN"]
        client = WebClient(token=slack_token)
        response = client.chat_postMessage(channel=channel, text=text)
        print("Message sent to slack: {}".format(response))
        return True
    except SlackApiError as e:
        print("There was an error sending the message to slack: {}".format(e))
        return False


def create_message_for_slack(
    output_json_file_path: str, input_pdf_file_path: str
) -> str:
    response = dedent(
        """\
            Here are the answers to your questions from the PDF file {input_pdf_file_name}:

            {output_question_answers}
        """
    )

    with open(output_json_file_path, "r") as f:
        question_answer_pairs = json.load(f)

    qa_text: str = ""
    for qa_pair in question_answer_pairs:
        qa_text += f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}\n\n"

    return response.format(
        input_pdf_file_name=input_pdf_file_path.split("/")[-1],
        output_question_answers=qa_text,
    )
